#!/usr/bin/env python3
"""
Per-sentence mmhal grader for caption-style transcripts.

For each record's caption response, splits into sentences and asks the judge
LLM to score EACH sentence 0-6 individually with the same MMHal rubric.
This gives a finer-grained hallucination signal than the per-round score
(which graded the whole caption with a single 0-6 verdict).

Usage:
  python grader/mmhal/mmhal_grader_per_sentence.py \\
      --response work_dirs/vg/caption/llava-1.5-7b-hf.json \\
      --evaluation work_dirs/vg/caption/llava-1.5-7b-hf/llava-1.5-7b-hf/mmhal_per_sentence_llava-1.5-7b-hf.json \\
      --gpt-model Qwen/Qwen3-30B-A3B-Instruct-2507 \\
      --api-url http://109.61.17.115:8000/v1/chat/completions \\
      --api-key william \\
      --gt-type caption \\
      --first-n 100 \\
      --workers 32
"""
import argparse, json, os, re, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _dir)
# Reuse the mmhal_grader template + helpers
from mmhal_grader import template, score_parsing_template, format_image_content_simple


def _split_sentences(text: str):
    """Split a caption response into sentences. Robust to lowercase captions."""
    if not text:
        return []
    # Normalize whitespace
    s = re.sub(r'\s+', ' ', text.strip())
    # Split on sentence terminators followed by whitespace. Don't require uppercase
    # after — many model outputs are all-lowercase. Common abbreviations like
    # "e.g.", "i.e.", "Mr.", "Dr." mostly don't show up in these captions, but if
    # they do they'll just produce slightly more, mostly-coherent fragments.
    parts = re.split(r'(?<=[.!?])\s+', s)
    # Also split on bullet/markdown patterns LLMs use (e.g. **header:**)
    extra = []
    for p in parts:
        # Split on lines starting with markdown bullets/asterisks-as-headers
        # that didn't end in punctuation
        sub = re.split(r'\s*\*\*[^*]+\*\*\s*[:.]?\s*', p)
        extra.extend(s2.strip() for s2 in sub if s2.strip())
    out = []
    for p in extra:
        # drop very short fragments (<3 words, no terminal punctuation)
        wc = len(p.split())
        if wc < 3 and not re.search(r'[.!?]$', p):
            continue
        out.append(p)
    return out


def _post_chat(api_url, api_key, model, prompt, max_retries=5, timeout=120, temperature=0.0, max_tokens=512):
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(min(2 ** attempt, 30))
            else:
                raise
    return None


def parse_score_with_llm(api_url, api_key, evaluation_text, gpt_model, max_retries=5, timeout=120):
    prompt = score_parsing_template.format(evaluation_text)
    try:
        content = _post_chat(api_url, api_key, gpt_model, prompt, max_retries=max_retries,
                             timeout=timeout, max_tokens=10)
        # Extract first digit 0-6
        m = re.search(r'[0-6]', content or "")
        return int(m.group()) if m else 0
    except Exception:
        return 0


def grade_sentence(api_url, api_key, model, image_content, prompt, gt, sentence, max_retries=5, timeout=120):
    eval_prompt = template.format(image_content, prompt, gt, sentence)
    try:
        eval_text = _post_chat(api_url, api_key, model, eval_prompt,
                               max_retries=max_retries, timeout=timeout, max_tokens=512)
    except Exception:
        return None, None
    if not eval_text:
        return None, None
    score = parse_score_with_llm(api_url, api_key, eval_text, model, max_retries=max_retries, timeout=timeout)
    return score, eval_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--response', type=str, required=True,
                   help='Input transcript JSON (e.g., work_dirs/vg/caption/<model>.json)')
    p.add_argument('--evaluation', type=str, required=True,
                   help='Output JSON with per-sentence scores')
    p.add_argument('--gpt-model', type=str, default='Qwen/Qwen3-30B-A3B-Instruct-2507')
    p.add_argument('--gt-type', type=str, default='caption', choices=['dyna', 'caption'])
    p.add_argument('--api-url', type=str, default=None)
    p.add_argument('--api-key', type=str, default=None)
    p.add_argument('--first-n', type=int, default=None)
    p.add_argument('--workers', type=int, default=32, help='parallel sentence-grading threads')
    args = p.parse_args()

    api_url = args.api_url or os.environ.get('REMOTE_API_URL') or 'https://api.openai.com/v1/chat/completions'
    api_key = args.api_key or os.environ.get('REMOTE_API_KEY') or os.environ.get('OPENAI_API_KEY')

    data = json.load(open(args.response))
    if args.first_n:
        data = data[:args.first_n]
    print(f"Records to process: {len(data)}", flush=True)

    # Build flat task list: one task per (record_idx, round_idx, sentence_idx)
    tasks = []  # (record_idx, round_idx, sent_idx, image_content, prompt, gt, sentence)
    for i, record in enumerate(data):
        # image content & gt mirror mmhal_grader.py logic
        if args.gt_type == 'caption':
            # gt = image info from metadata
            if 'metadata' in record:
                image_content = format_image_content_simple(record['metadata'])
            else:
                image_content = format_image_content_simple(record)
            gt_per_record = image_content
        else:
            if 'metadata' in record:
                image_content = format_image_content_simple(record['metadata'])
            else:
                image_content = format_image_content_simple(record)
            gt_per_record = None  # filled per-round below

        convs = record.get('conversations') or []
        for j, conv in enumerate(convs):
            prompt = conv.get('prompt', '')
            response = conv.get('response', '')
            gt = gt_per_record if args.gt_type == 'caption' else conv.get('gt', '')
            sentences = _split_sentences(response)
            for k, sent in enumerate(sentences):
                tasks.append((i, j, k, image_content, prompt, gt, sent))

    print(f"Total sentences to grade: {len(tasks)}", flush=True)

    # Parallel grading
    detailed = {}  # (i, j) -> list of (k, sent, score, eval_text)
    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for t in tasks:
            i, j, k, ic, pr, gt, sent = t
            fut = ex.submit(grade_sentence, api_url, api_key, args.gpt_model, ic, pr, gt, sent)
            futures[fut] = t
        for fut in as_completed(futures):
            t = futures[fut]
            i, j, k, ic, pr, gt, sent = t
            try:
                score, eval_text = fut.result()
            except Exception as e:
                score, eval_text = None, f"ERROR: {e}"
            detailed.setdefault((i, j), []).append({
                'sent_idx': k, 'sentence': sent, 'score': score, 'evaluation': eval_text
            })
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                print(f"  graded {completed}/{len(tasks)} sentences", flush=True)

    # Build output: per-record per-round breakdown + aggregated stats
    out_records = []
    all_scores = []
    record_means = []
    for (i, j), sents in sorted(detailed.items()):
        sents = sorted(sents, key=lambda x: x['sent_idx'])
        scores = [s['score'] for s in sents if s['score'] is not None]
        if scores:
            rmean = sum(scores) / len(scores)
            record_means.append(rmean)
            all_scores.extend(scores)
        out_records.append({
            'record_index': i,
            'round_id': j,
            'n_sentences': len(sents),
            'sentence_scores': sents,
            'sentence_avg_score': (sum(scores) / len(scores)) if scores else None,
        })

    overall = {
        'avg_score_per_sentence':       (sum(all_scores) / len(all_scores)) if all_scores else None,
        'avg_score_per_record':         (sum(record_means) / len(record_means)) if record_means else None,
        'mmhal_hallucination_per_sent': ((6 - sum(all_scores)/len(all_scores)) / 6 * 100) if all_scores else None,
        'n_sentences':                  len(all_scores),
        'n_records':                    len(record_means),
    }

    os.makedirs(os.path.dirname(args.evaluation), exist_ok=True)
    json.dump({'overall_metrics': overall, 'detailed_results': out_records},
              open(args.evaluation, 'w'), indent=2)
    print(f"\n=== overall ===", flush=True)
    for k, v in overall.items():
        print(f"  {k}: {v}")
    print(f"\nSaved -> {args.evaluation}")


if __name__ == '__main__':
    main()
