# UNet3D-Binary-Task
This is a UNet3D model designed for medical image segmentation.

Note: Most of this was AI-assisted, but I have manually debugged everything, so it is safe to use.

This code is primarily developed for tumor analysis in medical images.

It is available in both Traditional Chinese and English versions, with detailed information included.

1. Added optimizers (Adam, SGD, AdamW).
2. Added features such as momentum and warmup.
3. Added visualization functionality.
4. Early stopping mechanism (step, reduce_on_plateau, cosine).
5. Added the ability to perform data augmentation directly on the training set.

-------------------------------------------------------------------------------------------------------

這是一個專為醫學影像分割設計的 UNet3D 模型。

注意：此程式大部分由 AI 協助完成，但我已手動除錯，確保可以安全使用。

此程式碼主要用於醫學影像中的腫瘤分析。

提供繁體中文與英文版本，並包含詳細資訊。

1. 新增優化器（Adam、SGD、AdamW）。
2. 新增功能：momentum 與 warmup。
3. 新增視覺化功能。
4. 早停機制（step、reduce_on_plateau、cosine）。
5. 新增可直接在訓練集上進行資料增強的功能。

-------------------------------------------------------------------------------------------------------

Prediction Results and GT
<img width="1012" height="563" alt="image" src="https://github.com/user-attachments/assets/8c2b6f81-1374-4b05-9e3f-67368aeb42b5" />

-------------------------------------------------------------------------------------------------------

Visualization
<img width="1314" height="648" alt="image" src="https://github.com/user-attachments/assets/39ec9a77-96fa-4b4c-94e4-297d1e89ed04" />

-------------------------------------------------------------------------------------------------------

Final Results Display

<img width="578" height="551" alt="image" src="https://github.com/user-attachments/assets/36a140bb-ee4e-426d-a078-91bc17e50b03" />
<img width="639" height="694" alt="image" src="https://github.com/user-attachments/assets/8b0ada27-446e-4031-9f7a-2b8620c1b683" />








