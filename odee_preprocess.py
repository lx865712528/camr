import codecs
import json
import os
import sys

import constants
from amr_parsing import write_parsed_amr
from depparser import CharniakParser
from model import Model
from parser import Parser
from preprocessing import preprocess, StanfordCoreNLP


# for step1
def input_queue(INPUT_DIR):
    for DATE_STR in os.listdir(INPUT_DIR):
        if DATE_STR.startswith("."): continue
        for ID in os.listdir(os.path.join(INPUT_DIR, DATE_STR)):
            if ID.startswith("."): continue
            file_path = os.path.join(INPUT_DIR, DATE_STR, ID, "news.txt")
            if os.path.exists(file_path):
                yield file_path


def process_it(FILE_PATH, TEXT_MAXLEN):
    texts = []
    name = "-".join(FILE_PATH.split(os.sep)[-3:-1])
    flag = False
    with codecs.open(FILE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line.startswith("##"):
                flag = True
                continue
            if flag:  # skip all timestamps
                flag = False
                continue
            if len(line) <= TEXT_MAXLEN:
                texts.append(line)
    return [name], [texts]


# for step 2
def preprocess_stanford_and_camr(INPUT_DIR, MODEL_FILE):
    constants.FLAG_ONTO = "wsj"
    constants.FLAG_DEPPARSER = "stdconv+charniak"
    print("Loading model: ", MODEL_FILE)
    annotator = StanfordCoreNLP()
    annotator.setup()
    dparser = CharniakParser()
    model = Model.load_model(MODEL_FILE)
    parser = Parser(model=model,
                    oracle_type=constants.DET_T2G_ORACLE_ABT,
                    action_type="basic",
                    verbose=0,
                    elog=sys.stdout)
    print("Load all mdoel done!")

    for single_file in os.listdir(INPUT_DIR):
        if not single_file.endswith(".txt"):
            continue
        single_file = os.path.join(INPUT_DIR, single_file)
        print("Processing %s" % single_file)
        # stanford corenlp
        preprocess(single_file,
                   START_SNLP=True,
                   INPUT_AMR="sent",
                   PRP_FORMAT="plain",
                   proc1=annotator,
                   dparser=dparser)
        # camr
        test_instances = preprocess(single_file,
                                    START_SNLP=False,
                                    INPUT_AMR="sent",
                                    PRP_FORMAT="plain")
        span_graph_pairs, results = parser.parse_corpus_test(test_instances)
        parsed_suffix = '%s.%s.parsed' % ("all", "model")
        write_parsed_amr(results, test_instances, single_file, suffix=parsed_suffix)


# for step 3
def convert_to_sentences(obj_sentences):
    sentences = []
    for obj_sentence in obj_sentences:
        # list and dict
        sentence = []
        for token in obj_sentence.tokens:
            if token["id"] == 0: continue
            token["id"] -= 1
            sentence.append(token)
        # sorted
        sentences.append(list(sorted(sentence, key=lambda x: x["id"])))
    return sentences


def find_head(lemmas):
    for ans, lemma in enumerate(lemmas):
        if lemma in ["of", "that", "which", "by"]:
            if ans == 0:
                continue
            return ans - 1
    return len(lemmas) - 1


def convert_to_entities(json_sentences):
    entities = []
    for sid, sentence in enumerate(json_sentences):
        previous_label = "O"
        previous_start = -1

        for token in sentence:
            index = token["id"]
            entity_label = token["ne"]
            # link together
            if entity_label == "O":
                if previous_label != "O":
                    eid = len(entities)
                    forms = [x["form"] for x in sentence[previous_start:index]]
                    lemmas = [x["lemma"].lower() for x in sentence[previous_start:index]]
                    head_index = find_head(lemmas)
                    entities.append({
                        "id": eid,
                        "entity_Type": previous_label,
                        "startIndex": previous_start,
                        "endIndex": index,
                        "text": forms,
                        "headIndex": head_index + previous_start,
                        "sentNum": sid,
                        "headWord": forms[head_index],
                    })
                    previous_label = "O"
                    previous_start = -1
            else:
                if previous_label == "O":
                    previous_label = entity_label
                    previous_start = index
                elif entity_label == previous_label:
                    pass
                else:
                    eid = len(entities)
                    forms = [x["form"] for x in sentence[previous_start:index]]
                    lemmas = [x["lemma"].lower() for x in sentence[previous_start:index]]
                    head_index = find_head(lemmas)
                    entities.append({
                        "id": eid,
                        "entity_Type": previous_label,
                        "startIndex": previous_start,
                        "endIndex": index,
                        "text": forms,
                        "headIndex": head_index + previous_start,
                        "sentNum": sid,
                        "headWord": forms[head_index],
                    })
                    previous_label = entity_label
                    previous_start = index

        if not previous_label == "O":
            eid = len(entities)
            forms = [x["form"] for x in sentence[previous_start:]]
            lemmas = [x["lemma"].lower() for x in sentence[previous_start:]]
            head_index = find_head(lemmas)
            entities.append({
                "id": eid,
                "entity_Type": previous_label,
                "startIndex": previous_start,
                "endIndex": len(sentence),
                "text": forms,
                "headIndex": head_index + previous_start,
                "sentNum": sid,
                "headWord": forms[head_index],
            })
    return entities


def convert_to_corefs(json_entities):
    corefs = {}
    for entity in json_entities:
        head_word = entity["headWord"].lower()
        if head_word not in corefs:
            corefs[head_word] = []
        corefs[head_word].append(entity)
    return list(corefs.values())


class AMR:
    def __init__(self, lines):
        self.fathers = {}
        self.roles = {}
        self.attributes = {}

        previous_node = "0"
        previous_level = -1
        for line in lines:
            level = len(line) - len(line.lstrip())
            role = None
            node = None
            attribute = None
            line = line.rstrip(")")
            while len(line) > 0:
                line = line.lstrip()
                if line.startswith(":") and role is None:
                    rpos = line.find(" ")
                    role = line[1:rpos]  # extract ":ARG0"
                    line = line[rpos + 1:]
                elif line.startswith("(x") and node is None:
                    rpos = line.find(" ")
                    node = line[2:rpos]  # extract "x1"
                    line = line[rpos + 1:]
                elif line.startswith("/") or line.startswith("("):  # skip "/" or "(...."
                    rpos = line.find(" ")
                    line = line[rpos + 1:]
                elif attribute is None:
                    attribute = line
                    line = ""
            if node is None:
                continue
            while previous_level + 1 > level:
                previous_node = self.fathers[previous_node]
                previous_level -= 1
            self.fathers[node] = previous_node
            self.roles[node] = role if role is not None else ""
            self.attributes[node] = attribute if attribute is not None else ""
            previous_node = node
            previous_level = level

        self.remove_non_ints()

    def remove_non_ints(self):
        # skip all non integer nodes
        fathers, roles, attributes = {}, {}, {}
        for str_key, str_father in self.fathers.items():
            if not str_key.isdigit():
                continue
            while not str_father.isdigit():
                str_father = self.fathers[str_father]
            int_son = int(str_key) - 1
            fathers[int_son] = int(str_father) - 1
            roles[int_son] = self.roles[str_key]
            attributes[int_son] = self.attributes[str_key]
        self.fathers = fathers
        self.roles = roles
        self.attributes = attributes

    def __str__(self):
        return "graph is " + self.fathers.__str__() + "\n" + \
               "roles are " + self.roles.__str__() + "\n" + \
               "attrs are " + self.attributes.__str__()


def load_amr(fp):
    graphs = []
    amr_buffer = []
    for line in fp:
        line = line.rstrip()
        if line.startswith("#"):
            amr_buffer = []
            continue
        elif len(line) == 0:
            amr = AMR(amr_buffer)
            graphs.append(amr)
        else:
            amr_buffer.append(line)
    return graphs


def merge_json_amr(parsed_json, amr_graphs):
    # get a chain
    for coref in parsed_json["corefs"]:
        # get an entity mention
        for entity in coref:
            tokens = parsed_json["sentences"][entity["sentNum"]]
            graph = amr_graphs[entity["sentNum"]]
            start_index = entity["startIndex"]
            end_index = entity["endIndex"]
            head_index = entity["headIndex"]
            poss = [head_index] + list(range(start_index, end_index))
            # no predicate
            entity["predicate"] = ""
            entity["predicateFrame"] = ""
            entity["selfFrame"] = ""
            entity["predicate_relation"] = ""
            for pos in poss:
                # find a father
                if pos in graph.fathers:
                    father = graph.fathers[pos]
                    if father == -1:
                        father = pos
                    entity["predicate"] = tokens[father]["form"]
                    entity["predicateFrame"] = graph.attributes[father]
                    entity["selfFrame"] = graph.attributes[pos]
                    entity["predicate_relation"] = graph.roles[pos]
                    break


def generate_merge_json(INPUT_DIR):
    constants.FLAG_ONTO = "wsj"
    constants.FLAG_DEPPARSER = "stdconv+charniak"
    for fn in os.listdir(INPUT_DIR):
        if not fn.endswith(".txt"):
            continue
        single_file = os.path.join(INPUT_DIR, fn)
        print("Processing %s" % single_file)
        obj_sentences = preprocess(single_file,
                                   START_SNLP=False,
                                   INPUT_AMR="sent",
                                   PRP_FORMAT="plain")
        json_sentences = convert_to_sentences(obj_sentences)
        json_entities = convert_to_entities(json_sentences)
        json_corefs = convert_to_corefs(json_entities)
        parsed_json = {
            "sentences": json_sentences,
            "corefs": json_corefs,
        }
        with codecs.open(single_file + ".all.model.parsed", "r", encoding="utf-8") as f:
            amr_graphs = load_amr(f)
        merge_json_amr(parsed_json, amr_graphs)
        with codecs.open(single_file + ".json", "w", encoding="utf-8") as f:
            json.dump(parsed_json, f, ensure_ascii=False)


if __name__ == "__main__":
    # we must be under camr/
    if not os.path.abspath("odee_preprocess.py").split("/")[-2] == 'camr':
        print("We should be under camr/")
        sys.exit(-1)
    # get necessary arguments from CLI
    if len(sys.argv) != 6:
        print("PYTHON2 odee_preprocess.py ODEE_CORPUS_DIR OUTPUT_DIR AMR_MODEL ST_STEP ED_STEP")
        sys.exit(-1)
    ODEE_CORPUS_DIR, OUTPUT_DIR, AMR_MODEL, ST_STEP, ED_STEP = sys.argv[1], \
                                                               sys.argv[2], \
                                                               sys.argv[3], \
                                                               int(sys.argv[4]), \
                                                               int(sys.argv[5])
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR)

    # step 1 extract raw text
    if ST_STEP <= 1 and 1 <= ED_STEP:
        print("Step 1 start.")
        for file_path in input_queue(ODEE_CORPUS_DIR):
            print("Processing %s" % file_path)
            names, contents = process_it(file_path, TEXT_MAXLEN=600)
            for name, content in zip(names, contents):
                # dump raw texts
                raw_output_path = os.path.join(OUTPUT_DIR, name + ".txt")
                with codecs.open(raw_output_path, "w", encoding="utf-8") as f:
                    for line in content:
                        f.write(line.strip() + "\n")
        print("Step 1 done.\n\n\n\n")

    # step 2 run stanford and camr parser
    if ST_STEP <= 2 and 2 <= ED_STEP:
        print("Step 2 start.")
        preprocess_stanford_and_camr(OUTPUT_DIR, AMR_MODEL)
        print("Step 2 done.\n\n\n\n")

    # step 3 generate, clean and merge amr into json
    if ST_STEP <= 3 and 3 <= ED_STEP:
        print("Step 3 start.")
        generate_merge_json(OUTPUT_DIR)
        print("Step 3 done.\n\n\n\n")

    # finish
    print("Done!")
