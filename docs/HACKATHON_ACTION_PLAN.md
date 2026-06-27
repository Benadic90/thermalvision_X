# 🚀 THERMAVISION-X: 30-Hour Hackathon Action Plan
### Detailed Task Breakdown by Team Member & Phase
**Prepared by: Benad**

This document breaks down exactly what every member should be doing at every phase of the hackathon. Follow this step-by-step so nobody gets confused and no time is wasted.

---

## 👥 Meet the Team & Core Responsibilities

1. **Benadic (IT 1 - ML Lead)**: The PyTorch master. Focuses on the UNet architecture, the FFT Frequency Decoupling, and making sure the model trains without errors.
2. **Madhav (IT 2 - Data & Physics)**: The math and data pipeline builder. Focuses on loading satellite data, applying Planck's Law, and writing the physics-informed loss functions.
3. **Adity (IT 3 - Full-Stack, DevOps & Presenter)**: The integrator. Focuses on the Streamlit demo, exporting the model for edge deployment (ONNX), and managing the GitHub repository.
4. **Nisha (BMLT - Domain Expert)**: The storyteller. Focuses on finding real-world thermal images, preparing the pitch, writing medical/disaster use cases, and testing the final app.

---

## ⏱️ PHASE 1: Environment Setup & Data Prep (Hours 0-4)
*Goal: Everyone has a working environment, code is on GitHub, and data is ready.*

* **Benadic (IT 1 - ML Lead)**
  - [ ] Set up the Python environment (`pip install -r requirements.txt`).
  - [ ] Verify PyTorch runs with the RTX 2050 GPU.
  - [ ] Review `models/unet.py` and ensure the basic network compiles.
* **Madhav (IT 2 - Data & Physics)**
  - [ ] Download a visible RGB dataset (like Landsat visible bands or generic RGB images) into `assets/samples/visible` for training.
  - [ ] Download sample grayscale IR images (KAIST or FLIR) into `assets/samples/ir` for testing.
  - [ ] Run `data/ingestion.py` and `data/preprocessing.py` to ensure images load correctly.
* **Adity (IT 3 - Full-Stack & Presenter)**
  - [ ] Initialize the GitHub repository and push the `thermavision_x` folder structure.
  - [ ] Ensure all team members have access to the repo.
  - [ ] Run `streamlit run demo/app.py` to verify the UI loads locally.
* **Nisha (BMLT - Domain Expert)**
  - [ ] Read the `TEAM_GUIDE.md` and `THERMAVISION_X_PITCH.md`.
  - [ ] Search the web for compelling "Before (grayscale) & After (color)" thermal images in medicine (breast cancer, inflammation) to use in the pitch.

---

## ⏱️ PHASE 2: Core Model Building (Hours 4-12)
*Goal: The zero-shot pipeline is fully functional (FFT → UNet).*

* **Benadic (IT 1 - ML Lead)**
  - [ ] Implement and test the FFT mask in `models/frequency.py`.
  - [ ] Connect the FFT output (high frequencies) to the input of the UNet.
  - [ ] Run `pytest tests/test_frequency.py` to verify the math is correct.
* **Madhav (IT 2 - Data & Physics)**
  - [ ] Build the PyTorch DataLoader in `data/datasets.py`.
  - [ ] Ensure data augmentation (flips, rotations) is working.
  - [ ] Pass dummy batches to Benadic to test the model forward pass.
* **Adity (IT 3 - Full-Stack & Presenter)**
  - [ ] Build out the sidebar controls in the Streamlit app.
  - [ ] Add a file uploader that can handle `.tif` (satellite) and `.png` (standard) images.
  - [ ] Set up a dummy function that returns a fake colorized image when a user uploads an image, just so the UI flow is complete.
* **Nisha (BMLT - Domain Expert)**
  - [ ] Start drafting the actual PowerPoint/Canva presentation based on `THERMAVISION_X_PITCH.md`.
  - [ ] Create the "Problem Slide": Why grayscale is bad for human interpretation.

---

## ⏱️ PHASE 3: Physics Integration & Training (Hours 12-20)
*Goal: Loss functions are built, and the model starts training.*

* **Benadic (IT 1 - ML Lead)**
  - [ ] Write the main training loop in `train.py`.
  - [ ] Connect Madhav's loss function to the optimizer.
  - [ ] START THE TRAINING on the RTX 2050 (train on visible images only). Monitor for loss drops.
* **Madhav (IT 2 - Data & Physics)**
  - [ ] Implement Planck's Law and Stefan-Boltzmann constraints in `training/loss.py`.
  - [ ] Implement the structural similarity (SSIM) loss.
  - [ ] Verify the loss function doesn't throw `NaN` (Not a Number) errors during training.
* **Adity (IT 3 - Full-Stack & Presenter)**
  - [ ] Write the `inference/colorize.py` script.
  - [ ] Connect the Streamlit app to the inference script (so the app can use actual model weights once training is done).
* **Nisha (BMLT - Domain Expert)**
  - [ ] Write the "Solution Slide": Explain Zero-Shot learning and Physics-guidance in simple, non-technical words.
  - [ ] Practice explaining Planck's Law using the "Hot iron turning red" analogy.

---

## ⏱️ PHASE 4: App Demo & Pitch Polish (Hours 20-26)
*Goal: We have a trained model, it's plugged into the app, and the pitch is ready.*

* **Benadic (IT 1 - ML Lead)**
  - [ ] Stop training once the loss plateaus.
  - [ ] Implement the Monte Carlo Dropout for uncertainty maps in `models/uncertainty.py`.
  - [ ] Run tests to ensure the model correctly colorizes a grayscale IR image.
* **Madhav (IT 2 - Data & Physics)**
  - [ ] Help Benadic debug any weird artifacts in the colorized images (e.g., if colors look physically impossible, adjust the loss weights and retrain briefly).
  - [ ] Generate cool graphics showing the FFT separation (Image → Structure + Color) for the pitch deck.
* **Adity (IT 3 - Full-Stack & Presenter)**
  - [ ] Load the trained `best_model.pth` into the Streamlit app.
  - [ ] Test the app end-to-end: Upload image → Get colorized image + Confidence Map.
  - [ ] Fix any UI bugs or crashes.
* **Nisha (BMLT - Domain Expert)**
  - [ ] Take the Streamlit app and test it with 10 different images (medical, agriculture, disaster).
  - [ ] Save the best outputs to put into the presentation.
  - [ ] Finalize the pitch deck slides.

---

## ⏱️ PHASE 5: Deployment & Final Review (Hours 26-30)
*Goal: Export the model for edge devices, practice the pitch, and submit.*

* **Benadic (IT 1 - ML Lead)**
  - [ ] Clean up the codebase. Add comments where necessary.
  - [ ] Help Adity with the ONNX export if there are tensor shape issues.
* **Madhav (IT 2 - Data & Physics)**
  - [ ] Write a short `EVALUATION.md` file documenting the model's performance and physical accuracy.
* **Adity (IT 3 - Full-Stack & Presenter)**
  - [ ] Run `deployment/export.py` to convert the PyTorch model to an `.onnx` file.
  - [ ] Benchmark the `.onnx` file size and CPU inference speed to prove it runs on edge devices.
  - [ ] Submit the final GitHub repository link to the hackathon portal.
* **Nisha (BMLT - Domain Expert)**
  - [ ] Lead a full dry-run of the 5-minute pitch with the team.
  - [ ] Prepare answers for the FAQ (found in the Pitch doc).
  - [ ] Deliver the presentation to the judges!

---

## 🚨 Golden Rules for the 30 Hours
1. **Don't touch someone else's code without asking.** If Benadic is working on the UNet, Madhav should not edit it. Use GitHub properly.
2. **Commit often.** If something breaks at Hour 25, you want to be able to roll back to Hour 24.
3. **If training fails, fallback to simple.** If the Physics Loss is causing errors and you have only 2 hours left, disable it and just train with standard L1/L2 loss. A working simple model is better than a broken complex model.
4. **Adity is the boss of the pitch.** If she says an image looks confusing or a slide is too technical, the IT students must listen and simplify it. The judges are often not ML experts.
