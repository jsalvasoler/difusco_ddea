# Script to transfer the labeled graphs (1) to (0), and move the ones from (0) that have not been labeled to (1)

# Get labeled graphs from eowyn server
scp e12223411@eowyn.ac.tuwien.ac.at:/home1/e12223411/repos/difusco/data/mis/er_train_annotations_1.zip data/mis/
unzip data/mis/er_train_annotations_1.zip
ls data/mis/er_train_annotations_1/ | wc -l

# Move the labeled graphs from (1) to (0)
mv data/mis/er_train_annotations_1/* data/mis/er_train_annotations_0/
echo "Number of files in er_train_annotations_0: $(ls data/mis/er_train_annotations_0/ | wc -l)"
echo "Number of files in er_train_annotations_1: $(ls data/mis/er_train_annotations_1/ | wc -l)"


# empty er_train_1 and its zip
rm -rf data/mis/er_train_1/*
rm -rf data/mis/er_train_1.zip
rm -rf data/mis/er_train_annotations_1

# Move the unlabeled graphs from (0) to (1)
cd src/difusco_edward_sun/data/mis-benchmark-framework
python lil_trick.py --num_files 14811
cd ../../../..

# Print the number of files in each directory
echo "Number of files in er_train_annotations_0: $(ls data/mis/er_train_annotations_0/ | wc -l)"
echo "Number of files in er_train_annotations_1: $(ls data/mis/er_train_annotations_1/ | wc -l)"
echo "Number of files in er_train_1: $(ls data/mis/er_train_1/ | wc -l)"

# Finally zip the data/mis/er_train_1 file
zip -r data/mis/er_train_1.zip data/mis/er_train_1

# And send it to the eowyn server
scp data/mis/er_train_1.zip e12223411@eowyn.ac.tuwien.ac.at:/home1/e12223411/repos/difusco/data/mis/