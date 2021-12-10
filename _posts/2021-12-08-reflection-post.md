# Music Classifier - PIC16B Team Project

## The Github repository of our project

[repository](https://github.com/jiahao303/music-classifier)

## Overall, what did we achieve? (answer as a group)

I love music, and I always regard music as a source of stress relief. It is a great experience for me to do a project related to music. Music is easy to access nowadays thanks to technology development and the internet. However, when downloading music to our own phone, there is no genre classification, and getting all of the downloaded songs classified manually is hard. So, after our group's discussion, we finally decided to develop a model which would solve this problem and get the song’s genre classified from its metadata. Additionally, we would like to implement the model into a webapp, which is more accessible to users and, at the same time, gives a more straightforward visualization of how our model works.

## What am I especially proud of? 

* I am proud of how we utilized the Spotify Web API to transform the user input into the input for our model. After doing research on how to get the feature values (i.e. loudness, danceability, energy, etc.) of a song input by the user, I found an ingenious and helpful tool -- Spotify Web API. It is developed by Spotify and records the information including feature values of each song on Spotify. By writing a function which returns a dataframe with feature values of the input song, we successfully implemented the Spotify Web API. During the process of doing research and writing the function, I found myself learning to integrate the knowledge I already have had and the new information I just got from Google. From my point of view, continuously retrievinglearned knowledge and getting it applied brought me a greater sense of achievement than ever.

* Another thing that I am really proud of is how we finally implemented the model into our Webapp. Thanks to what we have learnt for text classification, we did not face many problems regarding building the model and got an 80% validation accuracy for the classification model. But we put in a great amount of time to work on implementing the model into the webapp. Since we are not very familiar with how to transform the user input to the type that a tensorflow model wants, we spent nearly two days trying to figure it out. Though the process of implementing the model was really tough, we finally debugged the program and got the model perfectly implemented.

## What would I suggest doing to further improve the project? 

* I think we could put more effort in improving the webapp functionally and aesthetically. 

> Aesthetics: Though we have worked on the aesthetic aspect of the webapp, but I am looking forward to more improvements. A beautiful and nicely designed interface can facilitate user experience, and thus adding more css style features to the webapp is necessary.

> Functionality: Our webapp has a textbox for the user to input the lyrics of a song. We did not discover a tool to help the webapp automatically link to lyrics with the info of the song name. So, the user has to input the lyrics value by googling on their own and copy/paste it. With this deficiency, the webapp would unlikely be evaluated as convenient.

* Another improvement would require a lot more time and effort. We could include a functionality that allows the user to give us the feedback of our prediction. To be specific, by collecting the user feedback, we could develop another dataset, which could potentially be used as training data to further improve our model in the future.

## How does what we achieved compare to what we set out to do in your proposal? (answer as a group)

* Our planned deliverables:
	
Compared to our plan, we have actually achieved almost all parts that we set out to do! We successfully build the genre classification model and create the web app interface. One slight difference is that we decided not to the include the sentiment analysis section but go with a more accurate and applicable way to construct the genre classification part by implementing the model to a mixed feature dataset (lyrics + scalar feature columns).

## What have I learned from the experience of completing this project? 

* **Python packages** I have learned more about tensorflow, especially in the field of text classification. Before this project, I had only done some basic machine learning with sklearn. But now, I am more adept with machine learning and deep learning. By learning more about machine learning, I personally felt the charm of it and of course, I realized that there are a lot more techniques for me to learn in machine learning.

* **Github!** Github is so helpful and powerful! I tried to work on Github before this class, and I felt confused about how to get things sorted on Github. Now, I am comfortable with using Github. On Github, we started our proposal, worked on our model code, and developed the app. Nothing about this project could be done without Github. Github allows us to commit new changes simultaneously. It is really effective and saves a lot of time.

* **Flask!** Flask is a totally novel thing for me. I had no idea about how people develop the interface before. And Flask is such a helpful tool in developing the interface. Actually, I felt a little bit exhausted when building the webapp at first since I am not familiar with Flask functions and syntax yet. But eventually, I got all things done at last. Though there are still lots of Flask features such as css style that I am not good at using, I am still proud of getting to know Flask, a powerful tool for developing the interface.

## How does my experience completing this project help my future studies or career?

By completing this project, I added something new related to data into my resume. Though the project is not perfect enough to go into the market right now and more improvements are needed, I believe what we have now has shown our skills and understanding of machine learning and webapp. This project deserves my discussion of what I have gained during the internship interview. 

More than the project itself, the most important thing I learnt is teamwork. To guarantee a basically smooth and pleasant experience of doing work projects, I think a group should be organized in regards to schedule planning, work division, and communication. In my opinion, we did not get these things clarified well at the beginning stage of the project, so we faced some problems when the project was in progress. Doing a group project will be very common in my future career, so I think having a deeper understanding of teamwork is really helpful to me. After reflecting on myself, I learnt that a regular group meeting is necessary to ensure that everybody is on the same page.



