Copyright (c) 2020 Colin Adams, Eli Weissler, Kai Kaneshina 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


# Colin_Eli_Kai_Math_178_Final

Repository with code for the math 178 final project, summer 2020

## Files

1) enviroment.yml: conda enviroment file to recreate conditions that the code ran in
2) To-Do.txt: our to do list. Much to do.

## Folders

**1) classifiers: code related to the classifier or feature-generation**

  * crossvalidated_classifier.py: Class for automating the training and evaluation of our classifier
  * featMatHelpers.py: Utility function for working with our feature matrices
  * features.py: Functions to generate hand-crafted features including torsion and curvature
  * linearRegression.py: Wrapper for performing linear regression easily
  * quaternions.py: Functions for performing rotations using quaternions
  * rotateByGyro: Function for performing our rotation and alignment by PCA
  
  
 **2) papers: pdfs of papers we read, along with some notes in a .txt**
 
 **3) prototyping: the poorly-written first attempts where we tested stuff**
 
  * skikit_supervised.ipynb: Notebook for prototyping the classifier, taken from sklearn example
  * neural_net.ipynb: Now abandoned attempt at recreating a published neural network for human activity recognition
  * Human Activity Recognition with Keras and CoreML original.ipynb: Another neural network test notebook
  * basic_quaternions.ipynb: Some basic quaternion operations
  * alignAccelerations.ipynb: Testing out the rotating and aligning scheme
  
  **4) readers: all the necessary data-wrangling**
   * createFeatVects.py: general code for creating feature vectors from a csv of individual observations
   * mobiActReader.py: code for assembling the mobiAct data from the file structure it comes in and for resampling to 50 hz. Also creates feature vectors for it.
   * motionsense_reader.py: code for assembling the motionsense data into a big csv of observations
   * UCI_HAR_reader.py: code for assembling the UCI-HAR data into a big csv of observations
   
  **5) visualizing_data: code for visualizing our data and creating figures**
   * fftVisualization.py: visualizing the power spectra of our acceleration data
   * orientation_and_rotation_plots.py: creating plots of the acceleration over time and of the PCA axes
   * pcaVisualization.py: creating plots of the PCA axes
   * sphere_plot.py: (very messy) code for plotting data on a sphere
   * spline_sphere_plot.py: code for plotting data on a sphere, with better interpolation
   * visualization/visualization_2.ipynd: general purpose visualiztion notebook for testing

