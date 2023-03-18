from hugsvision.dataio.VisionDataset import VisionDataset
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTImageProcessor

from hugsvision.inference.VisionClassifierInference import VisionClassifierInference

data_dir = 'mango_diseases'

train, test, id2label, label2id = VisionDataset.fromImageFolder(
    data_dir,
    test_ratio=0.15,
    balanced=True,
    augmentation=True,
)

huggingface_model = 'google/vit-base-patch16-224-in21k'

trainer = VisionClassifierTrainer(
    model_name="MyKvasirV2Model",
    train=train,
    test=test,
    output_dir="./out/",
    max_epochs=20,

    batch_size=32,  # On RTX 2080 Ti
    lr=2e-5,
    fp16=False,
    model=ViTForImageClassification.from_pretrained(
        huggingface_model,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label
    ),
    feature_extractor=ViTFeatureExtractor.from_pretrained(
        huggingface_model,
    ),
)

path = "./out/MyKvasirV2Model/20_2023-03-01-14-45-39/model" #choose path generated
img = "mango_diseases/Healthy/20211231_123105 (Custom).jpg"  # sample image path to test
model = ViTForImageClassification.from_pretrained(path)
feature_extractor = ViTImageProcessor.from_pretrained(path)
classifier = VisionClassifierInference(
    feature_extractor,
    model,
)

label = classifier.predict(img_path=img)
print("Predicted class:", label)