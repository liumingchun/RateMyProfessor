Three ".py" documents are the program and their raw results are '.csv'. Accordingly, '.xlsx' contain the charts from the '.csv' files.

[score_tb_result]
In the TextBlob Method, polarity suggesting the quantity of emotion ranges from -1 to 1 where -1 means the worse of the emotion and 1 means the best emotion. In the RMP case, when analyzing the relation between the score and the emotion of the comments regardless of the professor, we can find that besides the common expectation such as the positive relation of the score and the polarity, the polarity also fluctuates a lot when itself is lower than 0 and the score is lower than 2, meaning that the comments and score doesn't have  rigidly linear relation when the score is pretty low. For the subjectivity which is disciplined between 0 to 1, the statistic in this case comes to 0.58 generally. Also, it fluctuates a lot when the score is low.

[professor_tb_result]
Besides, for the relation between professor and the emotion evaluation of the comments, 0.072-0.315 is where the most students' sentimental score falls, suggesting the general postive attitude with the 0.51-0.663 subjectivity.

[professor_vader_result]
For the vader method,the standard to judge the emotion is setted for the following. If the compound is more than 0.5, the emotion is positive and if it is less than -0.5, the emotion is negative. Else, the emotion is neutral. With this standard, the positive comments are much more than negative and the neutrual ones. 