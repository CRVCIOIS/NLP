"""
Takes output from Gensim and returns most relevant NACE code(s)
"""
import json

def parse_codes(json_file):
    """
    Takes a json file of codes and returns a dictionary {description:code}
    
    :param json_file: a list of dictionaries
    :returns: a dictionary
    """

    with open(json_file, encoding="utf8") as json_file:
        data = json.load(json_file)
    
    desc_to_nace = {}

    for item in data:
        if item["Class"] is not None:
            nace = item['Section'] + item['Class']
            desc_to_nace[item['Activity']] = nace
    
    return desc_to_nace

def lookup_desc(data, kw):
    """
    Simple keyword lookup
    
    :param data: the desc_to_nace dictionary
    :param kw: the keyword we want to find
    :returns: a list of NACE codes
    """
    found_codes = []
    for desc in data.keys():
        if kw in desc:
            if not _is_negated(desc, kw): # negated word: "non-", "except"
                found_codes.append(data[desc])

    return found_codes

def _is_negated(desc, kw):
    if f"non-{kw}" in desc:
        return True
    if f"except {kw}" in desc:
        return True
    return False

if __name__ == "__main__":
    desc_to_nace = parse_codes("codes.json")
    found_codes = lookup_desc(desc_to_nace, "forestry")
    print (found_codes)