# IA-project
## Introduction:

Autism Spectrum Disorder (ASD) is a developmental disorder that affects communication, social interaction, and behavior. Individuals with ASD often struggle to understand and interpret emotional cues from others, which can lead to difficulties in building relationships and functioning in social situations. Recent research has shown that artificial intelligence (AI) can be used to develop tools to assist individuals with ASD in detecting emotions in people around them. This project aims to develop an AI system that can detect emotions in individuals and translate them into sounds that can be heard through an implant, such as a cochlear implant.

## Methods:

The proposed system is based on two primary models. The first uses VGG16 to detect emotions through the mel spectrogram. The model has been trained on English and Spanish datasets, achieving an accuracy rate of approximately 70% in recent tests. To further improve the accuracy, a distributed system was developed to set up a Telegram bot for individuals to help increase the dataset. This would allow individuals to provide feedback on the accuracy of the system, leading to a more precise and comprehensive dataset.

The second part of the system uses DistilBERT and Whisper to translate speech to text. Fine-tuning is then applied to DistilBERT to detect the conveyed emotion. This model can be used to detect emotions in speech that is not captured by the first model. The system is designed to be robust and capable of detecting emotions in real-time, making it suitable for use in social situations.

As a proof-of-concept, a mapping of piano notes to emotions was conducted to assign sound and emotion. This allows the system to generate sounds based on the emotion conveyed, which can be heard through an implant. The mapping was developed based on research on music and emotions, as well as consultation with individuals with ASD to ensure the assigned emotions were appropriate.

## Conclusion:

The use of AI in developing tools for individuals with ASD has shown promising results in recent years. The proposed system can assist individuals with ASD in detecting emotions in people around them, leading to improved social functioning and relationship-building. The system is designed to be robust and capable of detecting emotions in real-time, making it suitable for use in social situations. Further research and development are necessary to improve the accuracy and usability of the system, but the results show that the use of AI can have a significant impact on the lives of individuals with ASD.

The project was carried out by Mario Rico, Laura Rivero, and Carlos March as part of the final assignment for the Samsung Innovation Campus course. The team worked together to develop the system, which involved extensive research, data collection, and testing. The project demonstrates the potential of AI to improve the lives of individuals with ASD and highlights the importance of collaboration and innovation in addressing complex social challenges.
