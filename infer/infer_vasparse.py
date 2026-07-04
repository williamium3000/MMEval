import argparse
import os
import torch
import tqdm
import numpy as np
import random
import torch.backends.cudnn as cudnn
from torchvision import transforms
from minigpt4.models import load_preprocess
from minigpt4.common.config import Config
from minigpt4.common.registry import registry

from minigpt4.sparse.LlamaModelPatching import llamamodel_patching
import json

from types import SimpleNamespace
from decoder_zoo.Woodpecker.vis_corrector import Corrector
from decoder_zoo.Woodpecker.config import woodpecker_args_dict
from decoder_zoo.VCD.vcd_utils.vcd_add_noise import add_diffusion_noise
from decoder_zoo.VASparse.patching_generate import generate  # vasparse_decoding
from decoder_zoo.VASparse.vasparse_decoding import _update_model_kwargs_for_vasparse_contrastive_decoding, vasparse_search_contrastive_decoding
from decoder_zoo.VASparse.vasparser import vasparse_assistant
from PIL import Image
from mplug_owl2.mm_utils import process_images

MODEL_EVAL_CONFIG_PATH = {
    "minigpt4": "infer/eval_configs/minigpt4_eval.yaml",
    "instructblip": "infer/eval_configs/instructblip_eval.yaml",
    "shikra": "infer/eval_configs/shikra_eval.yaml",
    "llava-1.5": "infer/eval_configs/llava-1.5_eval.yaml",
    "mplug-owl2": "infer/eval_configs/mplug-owl2_eval.yaml",
}

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
    "mplug-owl2": "USER: <|image|><question> ASSISTANT:",
}

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)

device = "cuda" if torch.cuda.is_available() else "cpu"

lm_early_exit_layers = [
    0,
    2,
    4,
    6,
    8,
    10,
    12,
    14,
    16,
    18,
    20,
    22,
    24,
    26,
    28,
    30,
    32,
]


def setup_seeds(config, seed):
    # seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def patching(model, vasparse_contrastive_decoding):
    setattr(model.llama_model, "generate", generate.__get__(model.llama_model))

    if vasparse_contrastive_decoding:
        from decoder_zoo.VASparse.patching_generate import generate as vcs_generation  # vasparse_decoding
        # from decoder_zoo.VASparse.vasparse_decoding_minigpt4 import \
        #     vasparse_search_contrastive_decoding as vasparse_decoding_minigpt4
        setattr(model.llama_model, "generate", vcs_generation.__get__(model.llama_model))
        # setattr(model.llama_model, 'vasparse_search_contrastive_decoding', vasparse_decoding_minigpt4.__get__(model.llama_model)) #vasparse_decoding
        setattr(model.llama_model, 'vasparse_search_contrastive_decoding',
                vasparse_search_contrastive_decoding.__get__(model.llama_model))  # vasparse_decoding
        setattr(model.llama_model, '_update_model_kwargs_for_vasparse_contrastive_decoding',
                _update_model_kwargs_for_vasparse_contrastive_decoding.__get__(model.llama_model))  # vasparse_decoding


def eval_model(args, model, vis_processors):
    raw_image = Image.open(args.image_file).convert('RGB')
    if args.model_name == "mplug-owl2":
        max_edge = max(raw_image.size)  # We recommand you to resize to squared image for BEST performance.
        image = raw_image.resize((max_edge, max_edge))
        image_tensor = process_images([image], model.image_processor)
        image = image_tensor.to(device, dtype=torch.float16)
    else:
        image = vis_processors["eval"](raw_image).unsqueeze(0)
        image = image.to(device)

    template = INSTRUCTION_TEMPLATE[args.model_name]
    qu = args.query
    qu = template.replace("<question>", qu)

    mature_layer = lm_early_exit_layers[-1]
    premature_layer = None
    candidate_premature_layers = lm_early_exit_layers[:-1]
    premature_layer_dist = {l: 0 for l in candidate_premature_layers}

    args.vasparse_assistant_helper.update_input(img_path=args.image_file, input_prompt=qu)

    image_cd = None

    if args.vcd_decoding:
        image_tensor_cd = add_diffusion_noise(image, args.noise_step)
        image_cd = (
            image_tensor_cd.unsqueeze(0).half().cuda()
            if image_tensor_cd is not None
            else None
        )
        cd_alpha = args.cd_alpha
        cd_beta = args.cd_beta
        print("image_cd", image_cd.shape)
        print(cd_alpha, cd_beta, args.noise_step)
        if args.model_name == "minigpt4":
            image_cd = image_cd.squeeze(0)

    if args.prefill_sparse:
        llamamodel_patching(model, args.model_config)
    else:
        llamamodel_patching(model, args.model_config, [])

    patching(model, args.vasparse_contrastive_decoding)
    findings_kwargs = {'vit_token_debug': False}

    findings_kwargs['mask_rate'] = args.mask_rate
    findings_kwargs['contrastive_rate'] = args.contrastive_rate
    findings_kwargs['candidate_premature_layers'] = None
    findings_kwargs['base_layer'] = 0
    findings_kwargs['mature_layer'] = 32
    findings_kwargs['base_contrastive_layer'] = 0
    findings_kwargs['DoLa_is_layer'] = args.DoLa_is_layer

    findings_kwargs['max_sentence_lenght'] = args.max_sentence_lenght
    findings_kwargs['sparse_kv_cache_rate'] = args.sparse_kv_cache_rate
    findings_kwargs['prompt_length_image_text'] = 623 #86

    if args.vasparse_contrastive_decoding:
        findings_kwargs['vasparse_contrastive_decoding'] = True  # vasparse_decoding
        findings_kwargs['vasparse_assistant_helper'] = args.vasparse_assistant_helper
        findings_kwargs['SparsePrefilling'] = True
        findings_kwargs['beam_size'] = args.num_beams

    with torch.inference_mode():
        with torch.no_grad():
            out = model.generate(
                {"image": norm(image), "prompt": qu, "img_path": args.image_file},
                use_nucleus_sampling=args.sample,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                output_attentions=True,
                premature_layer=premature_layer,
                candidate_premature_layers=candidate_premature_layers,
                mature_layer=mature_layer,
                beam_search=args.beam_search,
                dola_decoding=args.dola_decoding,
                opera_decoding=args.opera_decoding,
                vcd_decoding=args.vcd_decoding,
                halc_decoding=args.halc_decoding,
                # HALC
                halc_assistant=args.vasparse_assistant_helper,
                # OPERA
                key_position=None,
                scale_factor=args.scale_factor,
                threshold=args.threshold,
                num_attn_candidates=args.num_attn_candidates,
                penalty_weights=args.penalty_weights,
                # VCD
                images_cd=image_cd,
                cd_alpha=args.cd_alpha,
                cd_beta=args.cd_beta,
                # VASparse
                findings_kwargs=findings_kwargs
            )

    output_text = out[0]

    print("original output text", output_text)
    sentence_list = output_text.split(".")
    sentence_filter_list = []
    for sentence in sentence_list:
        if "unk" not in sentence:
            sentence_filter_list.append(sentence)
    output_text = ".".join(sentence_filter_list)

    print("decoder output text", output_text)
    if args.post_correction == "woodpecker":
        sample = {
            "img_path": args.image_file,
            "input_desc": output_text,
            "query": qu,
        }

        corrected_sample = corrector.correct(sample)
        output_text = corrected_sample["output"]
        print("corrected output_text", output_text)

    return output_text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train domain generalization (oracle)')
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--outfile', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument("--model", type=str, default="llava-1.5", help="model")
    parser.add_argument(
        "-d",
        "--decoder",
        type=str,
        default="vasparse_contrastive",
        choices=["greedy", "dola", "halc", "opera", "vcd", "vasparse_contrastive"],
        help="Decoding strategy to use. You can choose from 'greedy', 'dola', 'halc'. Default is 'greedy'.",
    )
    parser.add_argument(
        "-g", "--gpu-id", type=int, default=0, help="specify the gpu to load the model."
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    parser.add_argument(
        "--mask_rate",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--contrastive_rate",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--DoLa_is_layer",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "--sparse_kv_cache_rate",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--max_sentence_lenght",
        type=int,
        default=16,
    )
    parser.add_argument("-b", "--beam", type=int, default=1)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--scale_factor", type=float, default=50)
    parser.add_argument("--threshold", type=int, default=15)
    parser.add_argument("--num_attn_candidates", type=int, default=5)
    parser.add_argument("--penalty_weights", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("-m", "--max_new_tokens", type=int, default=64)
    parser.add_argument(
        "-k",
        "--k-candidate-num",
        type=int,
        default=2,
        help="specify the k candidate number for halc.",
    )
    parser.add_argument(
        "-p",
        "--post-correction",
        type=str,
        default=None,
        choices=["lure", "woodpecker"],
        help="Post correction method such as woodpecker, lure.",
    )
    parser.add_argument(
        "--debugger",
        type=int,
        default=0,
        help="0 print no debugging output; 1 only print hallucination correction; 2 print all the debugging output.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="dino",
        choices=["dino", "owlv2"],
        help="Detector type. Default is 'groundingdino'.",
    )
    parser.add_argument(
        "-e",
        "--expand-ratio",
        type=float,
        default=0.6,
        help="Expand ratio of growing contextual field.",
    )
    parser.add_argument(
        "--cd_alpha",
        type=float,
        default=1,
        help="Alpha param for VCD.",
    )
    parser.add_argument("--cd_beta", type=float, default=0.1, help="Beta param for VCD.")
    parser.add_argument("--noise_step", type=int, default=500, help="Noise step for VCD.")
    parser.add_argument("--box_threshold", type=float, default=0.45, help="Box threshold for DINO.")
    args = parser.parse_args()

    args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
    cfg = Config(args)

    seed = args.seed
    setup_seeds(cfg, seed)

    model_name = args.model
    decoding_strategy = args.decoder
    k_candidate_num = args.k_candidate_num
    detector_type = args.detector
    num_beams = args.beam
    post_correction = args.post_correction
    max_new_tokens = args.max_new_tokens
    expand_ratio = args.expand_ratio
    cd_alpha = args.cd_alpha
    cd_beta = args.cd_beta
    box_threshold = args.box_threshold
    debugger = args.debugger

    halc_params = {
        "context_domain": "upper",
        "contrast_weight": 0.05,
        "context_window": 4,
        "expand_ratio": expand_ratio,
        "beam_size": num_beams,
        "k_candidate_num": k_candidate_num,
        "LVLM_backbone": model_name,
        "detector": detector_type,
        "score_type": "BLIP",
        "debugger": debugger,
        "box_threshold": box_threshold,
    }

    print("Initializing Model")
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(device)
    model.eval()

    processor_cfg = cfg.get_config().preprocess
    processor_cfg.vis_processor.eval.do_normalize = False
    vis_processors, txt_processors = load_preprocess(processor_cfg)

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
        vis_processor_cfg
    )

    print("decoding_strategy", decoding_strategy)
    opera_decoding = False
    dola_decoding = False
    halc_decoding = False
    vcd_decoding = False
    beam_search = False
    prefill_sparse = False
    vasparse_contrastive_decoding = False

    if decoding_strategy == "greedy":
        pass
    elif decoding_strategy == "dola":
        dola_decoding = True
    elif decoding_strategy == "halc":
        halc_decoding = True
        dola_decoding = True
        beam_search = True
    elif decoding_strategy == "opera":
        beam_search = True
        opera_decoding = True
    elif decoding_strategy == "vcd":
        vcd_decoding = True
    elif decoding_strategy == 'vasparse_contrastive':
        vasparse_contrastive_decoding = True
        prefill_sparse = True

    print('beam_search',beam_search)

    corrector = None
    if post_correction == "woodpecker":
        model_args = SimpleNamespace(**woodpecker_args_dict)
        corrector = Corrector(model_args)

    vasparse_assistant_helper = vasparse_assistant(
        model,
        vis_processor=vis_processor,
        device=device,
        halc_params=halc_params,
        max_new_tokens=max_new_tokens,
    )

    offlight = True

    samples = json.load(open(args.infile, "r"))
    for sample in tqdm.tqdm(samples):
        query = sample['question']
        image_file = os.path.join(args.img_dir, sample["image"])

        model_args = type('Args', (), {
            "model_name": model_name,
            "query": query,
            "image_file": image_file,
            "sample": args.sample,
            "cd_alpha": cd_alpha,
            "cd_beta": cd_beta,
            "prefill_sparse": prefill_sparse,
            "model_config": model_config,
            "num_beams": num_beams,
            "mask_rate": args.mask_rate,
            "contrastive_rate": args.contrastive_rate,
            "DoLa_is_layer": args.DoLa_is_layer,
            "max_sentence_lenght": args.max_sentence_lenght,
            "sparse_kv_cache_rate": args.sparse_kv_cache_rate,
            "max_new_tokens": max_new_tokens,
            "beam_search": beam_search,
            "noise_step": args.noise_step,
            "dola_decoding": dola_decoding,
            "opera_decoding": opera_decoding,
            "vcd_decoding": vcd_decoding,
            "halc_decoding": halc_decoding,
            "vasparse_assistant_helper": vasparse_assistant_helper,
            "scale_factor": args.scale_factor,
            "threshold": args.threshold,
            "num_attn_candidates": args.num_attn_candidates,
            "penalty_weights": args.penalty_weights,
            "corrector": corrector,
            "post_correction": post_correction,
            "vasparse_contrastive_decoding": vasparse_contrastive_decoding
        })()

        sample["output"] = eval_model(model_args, model, vis_processors)

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    json.dump(samples, open(args.outfile, "w"), indent=4)
