**I. Introduction**

  In recent years, the fast growing advancement of artificial intelligence and machine learning technologies has skyrocketed the potential of technology. Unfortunately this innovative path has brought a powerful and controversial concern to light, deepfakes. Deepfakes are synthetic media, including video, audio, or images, that are manipulated or generated using deep learning algorithms to effectively mimic real people. Forging these types of media are usually created using methods like Generative Adversarial Networks (GANs), which is a process of generating new data instances that are similar to a given dataset, making the difficulty of distinguishing authentic content ever growing.
  
  Although deepfakes do have potential positive uses in industries like entertainment, education, and accessibility, they also raise extreme concerns to the cyber security industry. Deepfakes can be exploited by malicious users to spread misinformation, impersonate individuals for fraud, manipulate public opinion, and compromise the integrity of communications sent digitally. People of high-profile status have already shown us how deepfakes can be weaponized for social engineering attacks, financial scams, and political propaganda.
  
  As deepfake technology continues to grow in its sophistication and accessibility, there is a dire need for robust detection practices. As synthetic media becomes even more realistic and pervasive, traditional verification tools like manual inspection and metadata analysis, are unsatisfactory. The importance of the development of reliable and proactive detection systems is imperative to protect the people, organizations, and societies from the evolving threats put forward by deepfakes.
  
  Even though there is ongoing research and development in deepfake detection, current models still display significant limitations in their effectiveness when applying them in real-world scenarios. In controlled environments, results are promising from current existing methods. But, there is still limited accuracy when exposed to high-quality deepfakes. This poor performance shows how vulnerable society is to attackers that are continuously refining their techniques in order to avoid detection.
  
  The computational demands that sufficient detection models require are hefty in processing power and memory. Because of this, scalability is difficult to accomplish as well as real-time usability, especially in IoT environments, or low bandwidth networks where resources are limited. The gap between researched performance and real-world application prove that current models are unreliable and impractical in settings where we must not forfeit accuracy.

  If these challenges remain unresolved, we as a society could face severe consequences. Deepfakes could strip away public trust in digital media, assist in identity theft, and provide ammunition to large-scale misinformation campaigns. In instances where highly sensitive data is involved, like government communications, financial transactions, and authentication systems, deepfakes that have gone undetected could lead to breaches of both privacy and trust. Therefore, getting rid of these gaps is not only a necessity but imperative to the wellbeing of our society.
  
  In this project, we aim to explore the effective of a deepfake detection system by training a custom classifier that uses a pre-trained EfficientNetB0 backbone and additional hand-crafted image features. Our system will prompt the user to upload a video, analyze facial features, regions, and various image statistics within the footage, and determine whether the content is authentic or has been manipulated. When a real face is detected, the program will visually confirm this by highlighting the individual with a green square, providing an intuitive form of feedback. 
  
  Due to limited computational resources and time constraints, we utilize a pre-trained EfficientNetB0 model as a feature extractor, but train a custom classifier on top of these features using our own dataset. Our focus lies in evaluating this model’s practical performance, identifying its strengths and weaknesses, and understanding how it behaves under different conditions. 
  
  Our project is guided by the following key research questions:
  - Can we achieve a high detection rate at a reasonable computational cost using a custom-trained classifier with a pre-trained EfficientNetB0 backbone and hybrid features in a lightweight pipeline?
  - How does our system compare with baseline approaches in terms of accuracy, speed, and usability?
  - By addressing these questions, we hope to contribute to the growing body of research aimed at making deepfake detection more accessible, efficient, and effective in real-world settings.

**II. Methods**

  The proposed deepfake detection system is designed for portability, ease of use, and robust performance on consumer-grade hardware. The architecture consists of three main stages: model loading, data processing, and result reporting.

  First, the system loads a pre-trained EfficientNetB0 model at runtime, using TensorFlow/Keras. The model weights are stored externally in an .h5 file and are loaded at the start of the program. This modular approach ensures that model updates or swaps can be performed without altering the main program logic. 

  Upon model loading, the system prompts the user to select a video file using a graphical file selection dialog powered by Tkinter. The selected video is read frame-by-frame using OpenCV, and each frame undergoes facial detection through the face_recognition library. For each detected face, the region of interest is extracted, resized to 224 x 224 pixels, normalized, and converted into a suitable input format for the neural network. In addition to the image, a set of custom features (such as sharpness, brightness, contrast, color metrics, and texture statistics) is extracted from the frame to enhance prediction robustness. The pre-processed face image is then passed to the EfficientNetB0-based model to predict the probability of manipulation. Processed frames are annotated with bounding boxes and classification (REAL or FAKE) and saved into an output video using OpenCV.

  Several aspects of this solution represent improvements over traditional deepfake detection pipelines:
- Simplified Deployment: The system operates entirely offline on Windows 10/11 with no external server setup or GPU requirements, facilitating easy deployment for non-expert users.
- Hybrid Feature Strategy: By combining deep features from EfficientNetB0 with hand-crafted image features, the system achieves higher reliability in detecting subtle manipulation. 
- Automated Post-Processing: The solution processes and annotates videos in a single step, streamlining the detection process for the user. 
  The system is developed entirely in Python 3.10, leveraging a range of specialized libraries:
- TensorFlow/Keras is used for loading and inferring with the EfficientNetB0 deep learning model.
- face_recognition is employed for efficient and accurate facial detection within video frames 
- OpenCV handles video capture, frame processing, and video writing tasks.
- Tkinter provides a lightweight graphical interface for video selection.
- numpy and tqdm are used for data processing and progress reporting

   In terms of optimization, the system emphasizes CPU-friendly design. EfficientNetB0’s relatively lightweight architecture ensures fast inference without GPU acceleration. Output videos are encoded using the mp4v codec, balancing quality and compression. Furthermore, the system caches frames efficiently and ensures directory management automatically to prevent processing interruptions.

  Although GPU acceleration is not utilized by default to prioritize portability, the system architecture remains compatible with TensorFlow-GPU should higher performance be required in future adaptations.

  For evaluation, we used the VideoTIMIT and DeepfakeTIMIT datasets, which provide paired real and deepfake video sequences of multiple subjects (Sanderson) (Korshunov and Marcel). These datasets offer a controlled yet diverse set of conditions for testing deepfake detection systems. 

  The system was evaluated on Windows 10/11 systems equipped with Intel Core i5/i7 processors without dedicated GPU support.

  The performance of the system was primarily assessed using accuracy, calculated as the proportion of correctly classified frames and videos. For video-level evaluation, the final prediction for each video was determined by averaging the frame-level scores and applying a threshold to classify the video as real or fake. Both frame-level and video-level predictions were analyzed. For video-level evaluation, an average of the final scores across all frames was used to derive a single classification decision.
  The codebase is structured for modularity and clarity. Major components include:
- A main driver script responsible for overall flow control, video I/O, and annotation.
- A configuration module for managing paths and settings
- Utility functions for face preprocessing, prediction, motion analysis, and frame annotation.
  Error handling is incorporated at critical points:
- Input validation ensures that users select valid video files.
- Defensive programming checks (e.g., verifying face crop dimensions) prevent runtime errors.
- Output directories are automatically created if absent, eliminating user setup errors.


  Security considerations are addressed by restricting file selection to .mp4 formats and minimizing reliance on user inputs, thereby reducing potential attack surfaces. The solution is designed with scalability in mind. It can be adapted to process large datasets by extending the file input logic to iterate over directories rather than single files. Frame caching and modularized processing also allow for parallelization, should higher throughput be required.

  Moreover, the architecture is readily adaptable for real-time environments. By employing multi-threading or asynchronous processing, it could be connected to live video feeds such as webcams or video streams. Integration with GPU acceleration via TensorFlow-GPU would further enable real-time deepfake detection at scale.

  Future directions include the addition of more advanced models, expansion to non-MP4 video formats, and the development of a user-friendly graphical user interface for broader accessibility.

**III. Results**

  The implementation of our deepfake detection system gave us mixed but insightful results regarding both feasibility and performance. While the modular design and lightweight architecture allowed the system to run effectively on consumer-grade hardware, several limitations became evident during testing.
 
  First, training a custom classifier for deepfake detection was feasible on standard student-level computers, though processing large video datasets and running inference on many frames remained time-consuming. Attempting to train larger detection networks or deepfake generation models locally was not viable due to insufficient memory, prolonged processing times, and overheating risks on non-specialized hardware. 

  Furthermore, while the use of a pre-trained EfficientNetB0 backbone provided a functional starting point, detection accuracy was far from perfect. The hybrid approach incorporating motion analysis and pixel variation did improve detection reliability for certain types of manipulated content, especially at lower resolutions. However, the system consistently struggled with high-quality or subtly altered deepfakes. Frame-level classification often exhibited inconsistencies, and the video-level predictions sometimes misclassified real content due to lighting artifacts or occluded faces.
 
  Overall, the results conclude with extreme difficulty in reliably identifying synthetic media using lightweight tools. While our pipeline demonstrated the potential for practical detection in constrained environments, it also highlighted the necessity of more powerful computational resources and advanced models to achieve consistently high accuracy.

**IV. Conclusion**

  This project spotlights both the promise and the current limitations of deepfake detection technologies in accessible, real-world contexts. Through the development and evaluation of a modular detection system built on a custom-trained classifier with a pre-trained EfficientNetB0 backbone, we demonstrated that basic detection is achievable on standard consumer hardware. However, our experience also exposed the significant computational demands and limitations in accuracy that accompany such efforts.
 
  The inability to train or fine-tune models efficiently without high-performance computing resources presents a barrier for many independent researchers or small organizations. Moreover, even with optimizations, our system was unable to deliver 100% reliable results, especially when faced with high-quality or subtle manipulations. This limitation is concerning given the increasing sophistication of synthetic media and the real-world consequences associated with their misuse.
  
  Ultimately, our findings stress the urgent need for continued research in creating more lightweight yet robust deepfake detection solutions. As deepfakes become more pervasive, democratizing access to effective detection tools will be critical in maintaining the integrity of digital communication. Future work should focus on expanding dataset diversity, incorporating more advanced neural architectures, and exploring scalable, real-time deployment strategies—ideally with support for GPU acceleration and parallel processing. 


Works Cited
Korshunov, Pavel, and Sebastien Marcel. “DeepfakeTIMIT.” Zenodo, 14 December 2018, https://zenodo.org/records/4068245. Accessed 5 May 2025.
Sanderson, Conrad. “The VideoTIMIT Database.” conradsanderson.id.au, University of Queensland, 2015, https://conradsanderson.id.au/vidtimit/. Accessed 5 May 2025.

