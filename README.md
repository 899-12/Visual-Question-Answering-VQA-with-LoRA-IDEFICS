# Visual-Question-Answering-VQA-with-LoRA-IDEFICS



This project fine-tunes the **IDEFICS-9B** vision-language model from Hugging Face on the **Flickr30k** dataset using **LoRA** (Low-Rank Adaptation) and **4-bit quantization** for the Visual Question Answering (VQA) task.

---

## 🚀 Project Highlights

- **Model**: [IDEFICS-9B](https://huggingface.co/HuggingFaceM4/idefics-9b)
- **Technique**: LoRA (Parameter-Efficient Fine-Tuning) + 4-bit Quantization (`bitsandbytes`)
- **Dataset**: [Flickr30k](https://huggingface.co/datasets/nlphuji/flickr30k) (1% sample)
- **Training Framework**: Hugging Face `Trainer`
- **Device Support**: CUDA / CPU fallback

---

## 📄 What This Project Does

- Loads the pre-trained IDEFICS-9B model with memory-efficient 4-bit quantization
- Preprocesses images and captions from the Flickr30k dataset
- Applies **LoRA** to tune only the essential attention parameters
- Trains and evaluates the model on image-description tasks
- Provides a caption generation function for inference

---

## 🏗️ Technologies Used

- Python
- PyTorch
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PEFT (LoRA)](https://github.com/huggingface/peft)
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)
- Hugging Face Datasets
- PIL, Torchvision

---

## 🧪 Sample Usage

```python
prompt = [
    image_url,
    "Describe this image."
]
generate_caption(model, processor, prompt)
