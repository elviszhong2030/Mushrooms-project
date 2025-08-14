#!/usr/bin/python3
import jetson_inference

import jetson_utils

import argparse
edible = [
    "Volvopluteus gloiocephalus",
    "Agaricus augustus",
    "Amanita amerirubescens",
    "Amanita calyptrodermsa",
    "Armillaria mellea",
    "Armillaria tabescens",
    "Artomyces pyxidatus",
    "Bolbitius titubans",
    "Boletus pallidus",
    "Boletus rex-veris",
    "Cantharellus californicus",
    "Cantharellus cinnabarinus",
    "Cerioporus squamosus",
    "Chlorophyllum brunneum",
    "Clitocybe nuda",
    "Coprinellus micaceus",
    "Coprinus comatus",
    "Flammulina velutipes",
    "Entoloma abortivum",
    "Ganoderma applanatum",
    "Ganoderma oregonense",
    "Grifola frondosa",
    "Hericium coralloides",
    "Hericium erinaceus",
    "Hypomyces lactifluorum",
    "Ischnoderma resinosum",
    "Laccaria ochropurpurea",
    "Lacrymaria lacrymabunda",
    "Lactarius indigo",
    "Laetiporus sulphureus",
    "Lycoperdon perlatum",
    "Lycoperdon pyriforme",
    "Mycena haematopus",
    "Pleurotus ostreatus",
    "Pleurotus pulmonarius",
    "Pluteus cervinus",
    "Psathyrella candolleana",
    "Pseudohydnum gelatinosum",
    "Psilocybe cyanescens",
    "Psilocybe muliercula",
    "Psilocybe pelliculosa",
    "Psilocybe zapotecorum",
    "Retiboletus ornatipes",
    "Sarcomyxa serotina",
    "Stropharia ambigua",
    "Stropharia rugosoannulata",
    "Suillus americanus",
    "Suillus luteus",
    "Suillus spraguei",
    "Tricholoma murrillianum"
]
non_edible = [
    "Tylopilus rubrobrunneus",
    "Tylopilus felleus",
    "Coprinopsis lagopus",
    "Crucibulum laeve",
    "Cryptoporus volvatus",
    "Fomitopsis mounceae",
    "Ganoderma curtisii",
    "Ganoderma tsugae",
    "Gliophorus psittacinus",
    "Gloeophyllum sepiarium",
    "Gymnopilus luteofolius",
    "Laricifomes officinalis",
    "Leucoagaricus americanus",
    "Leucoagaricus leucothites",
    "Lycogala epidendrum",
    "Mycena leaiana",
    "Panaeolus foenisecii",
    "Panellus stipticus",
    "Phaeolus schweinitzii",
    "Phyllotopsis nidulans",
    "Psilocybe caerulescens",
    "Psilocybe cubensis",
    "Psilocybe neoxalapensis",
    "Schizophyllum commune",
    "Stereum ostrea",
    "Tapinella atrotomentosa",
    "Trametes versicolor",
    "Trametes gibbosa",
    "Trametes betulina",
    "Trichaptum biforme",
    "Tricholomopsis rutilans",
    "Tubaria furfuracea"
]
toxic = [
    "Agaricus xanthodermus",
    "Amanita augusta",
    "Amanita brunnescens",
    "Amanita flavoconia",
    "Amanita muscaria",
    "Amanita persicina",
    "Amanita velosa",
    "Chlorophyllum molybdites",
    "Daedaleopsis confragosa",
    "Galerina marginata",
    "Hygrophoropsis aurantiaca",
    "Hypholoma fasciculare",
    "Hypholoma lateritium",
    "Leratiomyces ceres",
    "Omphalotus illudens",
    "Omphalotus olivascens",
    "Panaeolus cinctulus",
    "Panaeolus papilionaceus",
    "Phlebia tremellosa",
    "Psilocybe allenii",
    "Psilocybe azurescens",
    "Psilocybe aztecorum",
    "Psilocybe ovoideocystidiata"
]
parser = argparse.ArgumentParser()

parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("out_filename", type=str, help="filename of the image to output")
parser.add_argument("--model", type=str, default="resnet18.onnx", help="path to self-trained ONNX model to load")
parser.add_argument("--labels",type=str, default="labels.txt",help=" path to labels.txt file")
parser.add_argument("--input_blob",type=str, default="input_0")
parser.add_argument("--output_blob",type=str, default="output_0")

opt = parser.parse_args()
img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.imageNet(
    model=opt.model,
    labels=opt.labels,
    input_blob=opt.input_blob,
    output_blob=opt.output_blob,
 )
class_idx, confidence = net.Classify(img)
class_desc = net.GetClassDesc(class_idx)
print("image is recognized as "+ str(class_desc) +" (class #"+ str(class_idx) +") with " + str(confidence*100)+"% confidence")
result="not founded"
if class_desc in edible:
    result="edible"
elif class_desc in non_edible:
    result="non-edible"
elif class_desc in toxic:
    result="toxic"
font = jetson_utils.cudaFont(size=20)

font.OverlayText(
    img,
    text=f"Class: {class_desc} | Confidence: {confidence*100}%",
    x=0,
    y=0,
    color=(255, 255, 255, 255),
    background=(0, 0, 0, 120,)
)

font.OverlayText(
    img,
    text=f"It is a {result} mushroom",
    x=0,
    y=45,
    color=(255, 255, 255, 255),
    background=(0, 0, 0, 120,)
)
jetson_utils.saveImage(opt.out_filename, img)
print(f"Successfully classified image to {opt.out_filename}")
 