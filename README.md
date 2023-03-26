# Brain-tumor-segmentation
Brain MRI segmentation
Zeiad Ayman*, Yara Hossam El-Din *, Habiba Mohammed*
Inas Yassin *, Merna Atef*
*Department of System and Biomedical Engineering, Cairo University, Giza, Egypt  

## Abstract
In brain MRI analysis, image segmentation is commonly used for measuring and visualizing the brain's anatomical structures, for analyzing brain changes, for delineating pathological regions, and for surgical planning and image-guided interventions, Brain image segmentation is one of the most time-consuming and challenging procedures in a clinical environment. Since dealing with brain diseases is highly critical, medical staff usually need a second opinion, while conforming to the confidentiality of user privacy, AI systems propose a convenient workaround.Unet model was trained and used for this project which reached a dice coefficient of 0.79.
## Introduction
In the medical field, the analysis of medical images is crucial for the diagnosis and treatment of most modern diseases, especially life-threatening ones like tumors.
For the brain the safest option for imaging is MRI as it does not contain hazards of other imaging techniques like ionizing radiation in CT scan, also it gives the ability to see the organs in a whole new depth that isn't available by other means.
Due to the high risk of the disease and its consequences on the patient and his/her family, there is no room for error in such a diagnosis.Physicians should be highly trained in that regard in two aspects, the first is detecting the tumor whether its size is big or small, and the second is making sure that it’s, in fact, a tumor and not an enlarged part for example.This training process isn't easy and even if the physician is highly trained there is still room for error.
So, mostly such diagnoses are done by a team of physicians to ensure that it’s done correctly but this is not available in all treatment centers, so a computerized method to provide a second opinion for the physician is needed, and there is another reason for that which is the privacy of the patient.
If the patient is a high profile person information about his/her condition must be protected by all means as it may have ramifications that will affect him and those around him and even may affect society in some cases if the patient is in a high position like a government official for example.So, in such cases, these patients don't trust many physicians so a team opinion is out of option, and the physician needs to diagnose the patient correctly in such a situation a computerized solution will become very handy.

## literature review
we checked out what papers achieve until now, 
A. Evaluation of magnetic resonance image segmentation in brain low-grade gliomas using support vector machine and convolutional neural network [1]
Their method was based on SVM and CNN are developed and evaluated for brain LGG MR image segmentation studies. The segmentation performance in terms of accuracy and cost is quantitatively analyzed and compared between the SVM and CNN techniques developed.
And their result was SVM: avg acc.:0.937, Median acc.: 0.976, Precision: 0.456,0.535, Recall:0.878, 0.906, F1 Score: 0.546, 0.662. CNN: Acc:0.998, Precision:0.99, Recall:0.99, F1: .099
We conclude that CNN beats SVM in every possible    aspect, For segmentation Deep learning techniques reign supreme
B. Improved U-Net architecture with VGG-16 for brain tumor segmentation [2]
In this paper the methods they made were The Dense blocks in improved U-Net contribute towards feature mapping, which balances the vanishing gradient problem in convolutional blocks. The proposed modification (a) utilizes dense-convolutional blocks, instead of convolutional blocks during down-sampling, which enhances feature re-usability, (b) VGG16 pre-trained layers for encoder path and (c) Batch Normalization (BN) layers in dense blocks to improving model stability and performance.
Then their results UNet: Pixel acc.:0.994, Dice coefficient.:0.6829, Jaccard distance:0.5437 , Jaccard coefficient:0.5563. Improved UNet: Pixel acc.:0.9975
Dice cofficient.:0.9181, Jaccard distance:0.1736, Jaccard coefficient: 0.8264 
The conclusion was Using VGG as a decoder in an improved u net made a huge difference.
C. Learning Medical Image Denoising with Deep  Dynamic Residual Attention Network[3]
The proposed method is intended to recover the clean image from a given noisy medical image by learning residual noisy image through the mapping function.
The proposed network is presented as an end-to-end convolutional neural network (CNN). The network utilized traditional convolutional operations for input and output as well as a novel dynamic residual attention block (DRAB) as the backbone of the main network.
A total of 711,223 medical images were collected in this study. Where 585,198 samples were used for model training and the rest of the 20 percent data was used for performance evaluation.

The proposed method can exceed its counterpart on the dermatology images by up to 13.75 dB in the PSNR metric and 0.0992 in the SSIM metric.
But Novel technique in denoising images related to our dataset
D. Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm [4]
preoperative imaging and genomic data of 110 patients from 5 institutions with lower-grade gliomas from The Cancer Genome Atlas were used. Based on automatic deep learning segmentations, three features were extracted which quantify two-dimensional and three-dimensional characteristics of the tumors.
the result with the deep learning algorithm achieved a mean Dice coefficient of 82% which is comparable to human performance
the conclusion is a Low dice coefficient compared to other papers
Data
This data set called LGG Segmentation Dataset [5] contains brain MR images together with manual FLAIR abnormality segmentation masks.The images were obtained from The Cancer Imaging Archive (TCIA).They correspond to 110 patients included in The Cancer Genome Atlas (TCGA). Tumor genomic clusters and patient data are provided in the data.csv file.


## Methods
A.  Model Architecture
UNet was used with various shapes in this project, in general, UNet is a known deep learning network used for image segmentation and was created originally for medical image segmentation.
UNet can be divided into 3 parts which are the encoder, bottleneck, and decoder. The UNet also has skip connections or concatenations so that spatial info isn't lost after pooling layers, hence the network can classify the “what” and “where.”  Encoder's job is to understand info in the image, the decoder up-samples the image back. Figure.1 shows the UNet architecture.
In this project, various experimentations with hyperparameters were performed to see what will give the optimal performance. Two different UNet architectures were tested, the first one will be called the “1st  model” and the second is the” 2nd model.”

1st Model
The main building block of the encoder consisted of a conv2d layer followed by batch normalization and max pooling. Relu was used as the activation function.
The model contained 8 blocks with several filters starting from 32, 64, 128, 256 and then going down from 256, 128, 64, and 32 again. The model contained a total of 1,337,761 parameters.

Fig 1 :An illustration of the UNet model architecture
![alt text][Fig.1]

[Fig.1]:https://github.com/Zeyad-Ayman-Mohamed/Brain-tumor-segmentation/blob/main/images/BS_fig1.png 

2nd Model

The second model is another UNet with a contraction path block of two convolution layers followed by batch normalization and a max pooling layer. Relu was the activation function. 
The Contracting path has convolutions with an increasing number of filters per block as the following 64, 128, 256, 512, 1024. This gives the model the ability to learn the crucial features of the image. 
The expansion path then has the up-convolutions with the opposite effect to regain the spatial information of the image. The total model parameters are 31,043,521.

B. Data Splitting 
Many previous attempts to model this dataset used an odd way of splitting the data. They used slices of the same patient in training, validation, and testing sets. The model performed unnaturally well on a small number of epochs. This can be interpreted as model cheating. For a more robust approach, we split our data in terms of patients instead of slices. Each set has an exclusive group of patients. At first, we used consecutive folders in the same set. This wasn’t the best approach.
Then, we performed random splitting of patients among the three sets.

C. Data Augmentation
Given the small number of datasets, data augmentation was used to avoid overfitting with a combination of translation on the x and y axis with ranges of 0.05, zooming with a range of 0.05, horizontal flipping, and shear with a range of 0.05.

C. Hyperparameters Tuning
Hyperparameters tuning involved changing the learning rate, optimizer, batch size, and the number of iterations.

D. Model Evaluation Metrics
Since the mentioned problem is a segmentation one, using intersection over union (IOU) and dice coefficient was the most suitable choice to compare predicted masks with the ground truth.
IoU is the overlapping area between the predicted segmentation and the ground truth divided by the area of union between them, as shown in Fig. 2. This metric ranges from 0–1 (0–100%) with 0 signifying no overlap and 1 signifying perfectly overlapping.
The Dice Coefficient is 2 * the overlapping area divided by the total number of pixels in both images as shown in Fig.3.

Fig.2 shows the IOU equation
![alt text][Fig.2]

[Fig.2]:https://github.com/Zeyad-Ayman-Mohamed/Brain-tumor-segmentation/blob/main/images/BS_fig2.png 

Fig.3 shows the dice coefficient equation

![alt text][Fig.3]

[Fig.3]:https://github.com/Zeyad-Ayman-Mohamed/Brain-tumor-segmentation/blob/main/images/BS_fig3.png 
## results and discussion

A.  Model Architecture
1st Model
The first model showed an obvious case of underfitting. This can be interpreted as a result of the simplicity of the implementation and few trainable parameters. Even with running for 100 epochs and augmented data, the model couldn’t output any predictions. Dice loss and coefficient are shown in Fig.4.

Fig.4 shows the underfitting of the first model. The dice coefficient against iterations is shown at the top. The loss against iterations is shown at the bottom.
![alt text][Fig.4]

[Fig.4]:https://github.com/Zeyad-Ayman-Mohamed/Brain-tumor-segmentation/blob/main/images/BS_fig4.png 
2nd Model
The results of the second model were more promising but they still needed some improvement. However, this model was better suited for the problem as it didn’t underfit. The initial results of the second model after running on 40 epochs are shown in Fig.5.

Fig.5 shows the initial results of the second model on 40 epochs.  The dice coefficient against iterations is shown at the top. The loss against iterations is shown at the bottom.
![alt text][Fig.5]

[Fig.5]:https://github.com/Zeyad-Ayman-Mohamed/Brain-tumor-segmentation/blob/main/images/BS_fig5.png 
Fig.6 shows the initial results of the second model on 40 epochs. The original MRI image is on the left. The original mask is in the middle, and the predicted mask is on the right.
![alt text][Fig.6]

[Fig.6]:https://github.com/Zeyad-Ayman-Mohamed/Brain-tumor-segmentation/blob/main/images/BS_fig6.png 
B. Data Splitting Effects 
Splitting the data by patients made the model take much longer to reach good results but it also meant it was more robust to seeing completely new images. 
On splitting the patients’ folders consecutively, the testing patients were too hard for the patient to predict. This trial gave 0.7 training and validation dice and 0.4 testing dice coefficient.
To avoid this mismatch in performance, random splitting of the data was performed, this eliminated the gap in results. The initial results were 0.6 training dice and 0.5 testing dice on 50 epochs.


C. Data Augmentation
Performing data augmentation had the desired effect of reducing overfitting. There was no gap between training and testing dice coefficient anymore.

D. Hyperparameters Effect
Learning rate
Trying a learning rate of value 0.3 gave pretty bad results, this is probably because the learning rate was way too high for our problem, the model couldn’t get to the global minimum loss and it kept fluctuating, and going around the minima. 
Using a lower learning rate of 0.1 gave better results as the loss function stopped fluctuating and started converging. When applying the randomly split data, the model gave 0.84 training dice and 0.75 testing dice Plotting the loss against the number of iterations shows this convergence in Fig.7, and the predicted masks are shown in Fig. 8.
The graph still shows some fluctuations in the loss function, meaning that the learning rate could be lower for optimization purposes.
Finally, a learning rate of 0.01 gave the best results with 0.9 training dice and 0.79 testing dice. The graph of the loss function appeared smoother as shown in Fig.9. Predicted masks are shown in Fig. 10.
Optimizer
Experimenting with different optimizers gave different results. Adam optimizer gave the best performance with a final 0.49 testing dice compared with Adamax which gave 0.45 at 50 epochs for each. These results were before handling the overfitting problem due to biased data split.
Batch Size
Experimenting with different batch sizes showed the model tended to perform better on smaller batch sizes such as 32 and 15.
Number Of Epochs
Intuitively, given the small batch size, increasing the number of iterations enhanced model performance. 150 epochs gave the best results as the loss seemed to stay fixed after that.



Fig.7 shows the effect of random data splitting by patients with a 0.1 learning rate in the second model. The dice coefficient against iterations is shown at the top. The loss against iterations is shown at the bottom.
![alt text][Fig.7]

[Fig.7]:https://github.com/Zeyad-Ayman-Mohamed/Brain-tumor-segmentation/blob/main/images/BS_fig7.png 

Fig.8 shows the effect of random data splitting by patients with a 0.1 learning rate in the second model.  The original MRI image is on the left. The original mask is in the middle, and the predicted mask is on the right.
![alt text][Fig.8]

[Fig.8]:https://github.com/Zeyad-Ayman-Mohamed/Brain-tumor-segmentation/blob/main/images/BS_fig8.png 




Fig.9 shows the effect of random data splitting by patients with a 0.01 learning rate in the second model. The dice coefficient against iterations is shown at the top. The loss against iterations is shown at the bottom.

![alt text][Fig.9]

[Fig.9]:https://github.com/Zeyad-Ayman-Mohamed/Brain-tumor-segmentation/blob/main/images/BS_fig9.png 
Fig.10 shows the effect of random data splitting by patients with a 0.1 learning rate in the second model.  The original MRI image is on the left. The original mask is in the middle, and the predicted mask is on the right.
![alt text][Fig.10]

[Fig.10]:https://github.com/Zeyad-Ayman-Mohamed/Brain-tumor-segmentation/blob/main/images/BS_fig10.png 
## conclusion
Medical image segmentation saves radiologists intensive manual labor. Creating a robust model means following critical procedures to stop the model from cheating. This can be through correct data splitting techniques such as splitting data by patients and good distribution of the data among training and testing. Data augmentation is also crucial to handle small datasets, especially in medical image problems. UNet is the great start to a segmentation problem with medical images. With the right architecture, Adam optimizer, 0.01 learning rate, and 15 batch size, the 2nd model could achieve a dice coefficient of 0.8 after 150 epochs.








