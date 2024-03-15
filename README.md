#### This repository was established as a case study for a potential Junior/Middle DS position.
#### This material may not be copied or published elsewhere (including Facebook and other social media) without the permission of the author!


## Data Description & Preparation


• Non-Default: The loan is deemed non-defaulted if its delinquency status is below
90 days throughout the 24-month window

• Default: The loan is deemed defaulted if its delinquency status at least once
exceeds 90 days throughout the 24-month window.

The independent variables used in this study were selected from the origination dataset.
In contrast, the dependent variable was generated from the current loan delinquency status
variable in the monthly performance dataset based on the adopted definition of default. None
of the variables in the dataset have more than 50% missing values. In order to deal with the
missing values, a common approach was used where the missing values were replaced by their
mean value during data preparation. One-hot encoding was applied to categorical variables to
convert them into numerical dummy variables, enabling them to use in the modeling process.
The dataset was divided into a training set and a test set with 70% and 30% ratios, respectively.
The models were trained only on the training set, and their performance was evaluated on the
test set, which was an unseen dataset during the training process. 

**Author**:Adnan Sevinc

**Date**: 15.03.2024

**Version**: Draft

