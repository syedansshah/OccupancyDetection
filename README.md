# OccupancyDetection
The Occupancy Detection and Suspecious Activity detection
This is a project, to detect suspecious activity in the vacant room and human presence in the vacant room using sensor data.

The custom Dataset is also provided, which is taken in the university where there are classes going on in the rooms, so dataset has five fields;
Day	Room	StartTIme	EndTime	Course	Occupied

Day shows week's working days from Monday to Friday.
Room consist of Room name in the building/
StartTime is the time when sensor captured the human presence, is float value which start at 8.3
EndTime is the time when sensor captured no human presence, is float value which start at 10
Course is the course name which is being taught at startTime. it is string value.
Occupied in binary, 1 for occupied and 0 for not occupied.

Using this data, Machine Learning classification algo are trained and tested.
some of the algorithms used in the project are:

KNN (KNeighborsClassifier)
SVM (SVC or LinearSVC)
Naive Bayes (MultinomialNB)
Random Forest (RandomForestClassifier)
Decision Tree (DecisionTreeClassifier)
Neural Network (MLPClassifier)

REQUIREMENTS:
JUPYTER NOTEBOOK
PYTHON
DATASET

RELATED PAPERS:
[1] D. Marković, D. Vujičić, Z. Stamenkovič and S. Randič, "IoT Based Occupancy Detection System with Data Stream Processing and Artificial Neural Networks," 2020 23rd International Symposium on Design and Diagnostics of Electronic Circuits & Systems (DDECS), Novi Sad, Serbia, 2020, pp. 1-4, doi: 10.1109/DDECS50862.2020.9095715.
[2] Luis M. Candanedo, Véronique Feldheim, Accurate occupancy detection of an office room from light, temperature, humidity and CO2 measurements using statistical learning models,
Energy and Buildings, Volume 112, 2016,Pages 28-39, ISSN 0378-7788, https://doi.org/10.1016/j.enbuild.2015.11.071.

