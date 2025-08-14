# Mushrooms-project
Mushroom AI
 Mushroom AI is an AI - powered project that identifies mushrooms and classifies them as edible, non - edible, or toxic, helping users recognize mushrooms safely.

![Mushroom]( https://i.imgur.com/lhdedAy.jpeg "Mushroom")<br>
The Algorithm
The algorithm relies on Jetson inference’s imageNet and Resnet-18. First, it parses command - line arguments for the image file, output file, model, labels, and blob details. Then, it loads the input image and initializes an imageNet object with the specified model and labels.

The Classify function is used to get the class index and confidence score of the mushroom in the image. The images that are identified are downloaded from kaggle.The class description is retrieved with GetClassDesc.

There are three lists: edible (containing edible mushroom species), non_edible (non - edible species), and toxic (toxic species). The algorithm checks which list the class description belongs to and determines if the mushroom is edible, non - edible, or toxic.

Finally, text about the class, confidence, and edibility classification is overlaid on the image using cudaFont.OverlayText ,and the modified image is saved. This process enables automated mushroom identification and safety classification using pre - trained deep learning models.

Running this project

Install required libraries:

1.Install jetson_inference and jetson_utils. You can follow the installation instructions specific to your Jetson device (e.g., Jetson Orin Nano).

2.Ensure you have Python 3 installed.

Prepare files:

1.Download a dataset for pictures     (My dataset is from kaggle:   https://www.kaggle.com/datasets/iftekhar08/mo-106)

2.Have a labels.txt file with the labels of different types of mushrooms . 

3.Have a trained model file:							 (e.g., I use the resnet18 as my model: resnet18.onnx).

4.Prepare the python code to identify the pictures

Run the script:

1.Open the terminal.

2.Execute the script with command - line arguments specifying the input image filename and the output image filename,etc. For example: python3 my-recognition.py Entoloma_abortivum_66.jpg Mushrooms1.jpg

Here is the code:

```python
#!/usr/bin/python3
import jetson_inference


import jetson_utils


import argparse
edible=["Volvopluteus gloiocephalus", "Agaricus augustus", "Amanita amerirubescens", "Amanita calyptrodermsa", "Armillaria mellea", "Armillaria tabescens", "Artomyces pyxidatus", "Bolbitius titubans", "Boletus pallidus", "Boletus rex-veris",
        "Cantharellus californicus", "Cantharellus cinnabarinus", "Cerioporus squamosus", "Chlorophyllum brunneum", "Clitocybe nuda", "Coprinellus micaceus", "Coprinus comatus", "Flammulina velutipes", "Entoloma abortivum",
        "Ganoderma applanatum", "Ganoderma oregonense", "Grifola frondosa", "Hericium coralloides", "Hericium erinaceus", "Hypomyces lactifluorum", "Ischnoderma resinosum", "Laccaria ochropurpurea", "Lacrymaria lacrymabunda",
        "Lactarius indigo", "Laetiporus sulphureus", "Lycoperdon perlatum", "Lycoperdon pyriforme", "Mycena haematopus", "Pleurotus ostreatus", "Pleurotus pulmonarius", "Pluteus cervinus", "Psathyrella candolleana", "Pseudohydnum gelatinosum", "Psilocybe cyanescens", "Psilocybe muliercula", "Psilocybe pelliculosa",
        "Psilocybe zapotecorum", "Retiboletus ornatipes", "Sarcomyxa serotina", "Stropharia ambigua", "Stropharia rugosoannulata", "Suillus americanus", "Suillus luteus", "Suillus spraguei", "Tricholoma murrillianum"]
non_edible=["Tylopilus rubrobrunneus", "Tylopilus felleus", "Coprinopsis lagopus", "Crucibulum laeve", "Cryptoporus volvatus", "Fomitopsis mounceae", "Ganoderma curtisii", "Ganoderma tsugae", "Gliophorus psittacinus", "Gloeophyllum sepiarium",
            "Gymnopilus luteofolius", "Laricifomes officinalis", "Leucoagaricus americanus", "Leucoagaricus leucothites", "Lycogala epidendrum",
            "Mycena leaiana", "Panaeolus foenisecii", "Panellus stipticus", "Phaeolus schweinitzii", "Phyllotopsis nidulans", "Psilocybe caerulescens", "Psilocybe cubensis", "Psilocybe neoxalapensis",
            "Schizophyllum commune", "Stereum ostrea", "Tapinella atrotomentosa", "Trametes versicolor", "Trametes gibbosa", "Trametes betulina", "Trichaptum biforme", "Tricholomopsis rutilans", "Tubaria furfuracea"]
toxic=["Agaricus xanthodermus", "Amanita augusta", "Amanita brunnescens", "Amanita flavoconia", "Amanita muscaria", "Amanita persicina", "Amanita velosa", "Chlorophyllum molybdites", "Daedaleopsis confragosa", "Galerina marginata", "Hygrophoropsis aurantiaca", "Hypholoma fasciculare", "Hypholoma lateritium", "Leratiomyces ceres",
       "Omphalotus illudens", "Omphalotus olivascens", "Panaeolus cinctulus", "Panaeolus papilionaceus", "Phlebia tremellosa", "Psilocybe allenii", "Psilocybe azurescens", "Psilocybe aztecorum", "Psilocybe ovoideocystidiata"]
parser = argparse.ArgumentParser()


parser.add_argument("filename", type=str, help="filename of the image to process")
parser.add_argument("out_filename", type=str, help="filename of the image to output")
parser.add_argument("--model", type=str, default="resnet18.onnx", help="model to use, can be:  googlenet, resnet-18, ect. (see --help for others)")
parser.add_argument("--labels",type=str, default="labels.txt",help=" path to labels.txt")
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
 ```

