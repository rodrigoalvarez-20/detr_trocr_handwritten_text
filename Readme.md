### DETR & TROCR for handwritten text recognition

This repository contains work related to an investigation of the detection and recognition of handwritten text using transformers.

The main goal of the investigation is to create a novel form to detect and recognize handwritten text, in particular, cursive text.

To achieve this, I created a simple pipeline of **text detection** and **optical** **character** **recognition** using the Transformers architecture.

I used the SOTA models **DETR** (Detection-Transformers) and **TROCR** (Transformers for optical character recognition), finetuned both of them (separately) and concatenate the output from the DETR, using an intermediate step that consist in cropping the original image, with the TROCR input.

At the end of the pipeline, the TROCR gives to the user the recognized text.


Author: Rodrigo Alvarez

Year: 2024