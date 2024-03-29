##!/bin/sh

mkdir -p HAR
cd HAR
mkdir -p datasets
cd datasets

#REALDISP
echo -e "\n###############"
echo GETTING REALDISP DATA
echo -e "###############\n"
mkdir -p realdisp
cd realdisp
mkdir -p RawData
#wget archive.ics.uci.edu/ml/machine-learning-databases/00305/realistic_sensor_displacement.zip
wget archive.ics.uci.edu/static/public/305/realdisp+activity+recognition+dataset.zip
cd RawData
unzip realistic+sensor+displacement.zip
cd ../..
python ../convert_data_to_np.py REALDISP
cd ../..
