# C3AE-pytorch
C3AE:exploring the limits of compact model for age estimation 
just for fun!
imdb dataset preprocess:
1 mtcnn detection and landmark regression

2 too lazy... dont change detection code and not use detection information,according to landmarks,crop three scale faces;
  if landmarks is not complete,the picture will be dropped.
  
3 different from paper,drop out is not used.SE module or other losses do not implement

4 if have enough time ,the project will be completed.

5 split 30% Imdb as test data,on test set MAE is 7.51


