import os
import sng_parser
from dyna.utils import load_coco2017, get_synset

det_list = ['a', 'the', 'an', 'this', 'that', 'these', 'those', '']


def reformat_graph(graph):
    entities = graph['entities']
    relations = graph['relations']

    entities_data = {}
    for e in entities:
        entity_name = e['head'].lower()
        attribute_list = []
        for x in e['modifiers']:
            attribute = x['span'].lower()
            if attribute not in det_list:
                attribute_list.append(attribute)

        if entity_name not in entities_data:
            entities_data[entity_name] = attribute_list
        else:
            entities_data[entity_name] += attribute_list

    relations_data = [
        [
            entities[rel['subject']]['head'].lower(),
            rel['relation'].lower(),
            entities[rel['object']]['head'].lower()
        ]
        for rel in relations
    ]
    return entities_data, relations_data


if __name__ == "__main__":
    case = load_coco2017(324158)
    image_file = os.path.join("data/coco/val2017", case["file_name"])

    gt_caption = ' '.join(case["captions"])
    gt_graph = sng_parser.parse(gt_caption)
    gt_entities, gt_relations = reformat_graph(gt_graph)

    vlm_output = (
        "A person riding a skate board with a dog following beside. the image features a man riding a skateboard down a path, accompanied by his dog the man is wearing a "
        "green jacket and appears to be enjoying the outdoor activity the dog is walking beside him, "
        "keeping pace with the skateboarder the scene takes place in a park-like setting, with several cars "
        "parked along the path there are at least nine cars visible in the background, varying in size and "
        "distance from the skateboarder and his dog the overall atmosphere of the image is lively and active, "
        "showcasing the man and his dog engaging in a fun and healthy outdoor activity.")

    vlm_graph = sng_parser.parse(vlm_output)
    vlm_entities, vlm_relations = reformat_graph(vlm_graph)

    object_hallucination_num = 0
    object_uncertain_num = 0
    relation_hallucination_num = 0
    for object_name, attribute_list in vlm_entities.items():
        object_syn_list = get_synset(object_name)
        common_objects = set(object_syn_list).intersection(gt_entities.keys())
        if common_objects:
            for object_synonym in common_objects:
                gt_attribute_list = gt_entities[object_synonym]
                for attribute in attribute_list:
                    attribute_syn_list = get_synset(attribute)
                    common_attributes = set(attribute_syn_list).intersection(gt_attribute_list)
                    if not common_attributes:
                        object_hallucination_num += 1
                        print('Hallucination')
                        # object_uncertain_num += 1
                        # print('Uncertain')
                        print('vlm', object_name, attribute_list)
                        print('gt', object_name, gt_attribute_list)
                        print('\n')
                        break
        else:
            object_uncertain_num += 1
            print('Uncertain')
            print('vlm', object_name, attribute_list)
            print('\n')

    for relation in vlm_relations:
        if relation not in gt_relations:
            relation_hallucination_num += 1
            # TO DO
            # print('vlm relation', relation)

    print('vlm_graph')
    sng_parser.tprint(vlm_graph)
    print('gt_graph')
    sng_parser.tprint(gt_graph)

    print('object_hallucination_num:', object_hallucination_num)
    print('relation_hallucination_num:', relation_hallucination_num)
    print('object_uncertain_num:', object_uncertain_num)
