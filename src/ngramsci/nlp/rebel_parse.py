"""AGPL-3.0: copied from GenderedNews citations_masc_computing (Ange Richard), https://github.com/getalp/genderednews-tools

extract_triplets_rcv2 + rcv2 constants, copied verbatim from GenderedNews citations_masc_computing (Ange Richard)
https://github.com/getalp/genderednews-tools"""
class _R: pass
rcv2_dataset = _R()
_DESCRIPTION = """RCV2 is a dataset created for the paper """""
_URL = ""
_URLS = {
    "train": _URL + "en_train.jsonl",
    "dev": _URL + "en_val.jsonl",
    "test": _URL + "en_test.jsonl",
}
TAG2ENTITY = {
    '<per>': 'Agent',
    '<peop>': 'Group_of_people',
    '<pron>': 'Source_pronoun',
    '<org>': 'Organization',
    '<cue>': 'Cue',
    '<dirquot>': 'Direct_Quotation',
    '<indquot>': 'Indirect_Quotation',
    '<mixquot>': 'Mixed_Quotation'
    }
ENTITY2TAG = {v:k for k,v in TAG2ENTITY.items()}
REL_LABEL2TEXT = {
    'Quoted_in': 'cité',
    'Indicates': 'indique',
    #'Refers': 'référence'
}
REL_TEXT2LABEL = {v:k for k, v in REL_LABEL2TEXT.items()}
FR_REL_LABEL2TEXT = REL_LABEL2TEXT
FR_REL_TEXT2LABEL = REL_TEXT2LABEL
for _k, _v in list(globals().items()):
    if _k.isupper(): setattr(rcv2_dataset, _k, _v)

def extract_triplets_rcv2(text, mapping_types=None, build_data_for_mREBEL=False):
    # print(f"============ TEXT TO EXTRACT TRIPLETS FROM =============\n {text}")
    mapping_types = rcv2_dataset.TAG2ENTITY if not mapping_types else mapping_types
    REL_LABEL2TEXT = rcv2_dataset.EN_REL_LABEL2TEXT if build_data_for_mREBEL else rcv2_dataset.FR_REL_LABEL2TEXT

    def extract_relations(lst):
        # cut triplets based on the appearance of relation labels
        rels = []
        rel_items = []
        for item in (lst):
            rel_items.append(item)

            if item in REL_LABEL2TEXT.values():
                rels.append(rel_items)
                rel_items = []
        return rels

    # warn: special tokens are "pasted" to the preceding word by the tokenizer, so we add back the missing whitespace
    text = text.strip()
    word_list = text.replace('fr_XX', '').replace('tp_XX', '').replace('<', ' <').replace('<s>', '').replace('</s>',
                                                                                                             '').replace(
        '<pad>', '').split()
    if not word_list:  # some texts might be only a sequence of special characters so the word_list will be empty
        return []
    # print(f"text: {text}")
    # print(f"cleaned word list: {word_list}")

    # at the beggining of training, there might not be any or not enough tags/items in the prediction
    # to extract, so we make sure to handle these cases
    triplet_indices = [idx for idx, word in enumerate(word_list) if word == '<triplet>']
    # print(f"triplet indices: {triplet_indices}")

    triplet_list = []  # [[]] where each [] contains all the triplets from a doc word by word
    if triplet_indices:
        for item in reversed(triplet_indices):
            triplet_list.append(word_list[item:])
            word_list = word_list[:item]
        triplet_list = [t[1:] for t in triplet_list]  # remove <triplet> tag from each list
        triplet_list = list(reversed(triplet_list))
        # triplet_list = [' '.join(t) for t in triplet_list]
        # print(triplet_list)
    else:
        # no triplet tags were found, we assign the whole predicted text as triplet
        triplet_list.append(word_list)
    # print(f"triplet list: {triplet_list}")

    triplets = []  # [[]] where each [] contains the triplets item by item (either tags or full sentences)
    for t in triplet_list:
        items = []
        text = ''
        for idx, word in enumerate(t):
            if word not in mapping_types.keys() and idx != len(t) - 1:
                text += f'{word} '
            elif word not in mapping_types.keys() and idx == len(t) - 1:  # last word of the triplet
                text += f'{word} '
                items.append(text)
            elif word in mapping_types.keys() and text != '':
                items.extend([text, word])
                text = ''
        items = [w.strip() for w in items]
        triplets.append(items)
    # print(f"triplets: {triplets}")

    # we return the relations in the direction they were annotated (refer to .jsonl dataset files)
    # refer to paper section _ for details on the triplet structure extracted here
    relations = []

    # TODO: check what to do if empty triplets are generated
    if triplets:
        for triplet in triplets:
            # check if the correct number of tags/items per triplet were predicted, return empty strings if needed

            if len(triplet) > 1:
                tail = triplet[0]  # the tail will always be either a quote or a subject
                rels = extract_relations(triplet[1:])

                for rel in rels:
                    if len(rel) == 4:
                        relations.append(
                            {'head': rel[1], 'head_type': rel[2], 'tail': tail, 'tail_type': rel[0], 'type': rel[3]})
                    else:
                        relation = {'head': None, 'head_type': None, 'tail': tail, 'tail_type': None, 'type': None}
                        for item in rel:
                            if item in list(REL_LABEL2TEXT.values())[:3] or item in list(REL_LABEL2TEXT.values())[
                                                                                    5:] and not relation['tail_type']:
                                relation['tail_type'] = item
                            elif item in list(mapping_types.keys()) and not relation['head_type']:
                                relation['head_type'] = item
                            elif item in REL_LABEL2TEXT.values() and not relation['type']:
                                relation['type'] = item
                            elif item not in list(mapping_types.keys()) and not relation['head']:
                                relation['head'] = item

                        for k, v in relation.items():
                            if not v:
                                relation[k] = ''
                        relations.append(relation)

            elif len(triplet) == 1:
                # in the beg of training, BARThez REBEL sometimes predicted a single chunk of text with no labels
                relations.append({'head': '', 'head_type': '', 'tail': triplet[0], 'tail_type': '', 'type': ''})

    # print(f"relations: {relations}")
    return relations
