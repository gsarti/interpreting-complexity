export GECO=http://expsy.ugent.be/downloads/geco/files/MonolingualReadingData.xlsx
export GECO_EXTRA=http://expsy.ugent.be/downloads/geco/files/EnglishMaterial.xlsx

# Set this to true to download MAT files for the ZuCo v1 and v2 corpora
# Approximately 30GB to download due to EEG information 
export DOWNLOAD_ZUCO_MAT_FILES=false

export ZUCO_V1_NR_URL=https://files.osf.io/v1/resources/q3zws/providers/osfstorage/5b4eee6d33f0b9000cba7151/?zip=zuco1_nr.zip
export ZUCO_V1_SR_URL=https://files.osf.io/v1/resources/q3zws/providers/osfstorage/5b4eee366d4eb300106ec69c/?zip=zuco1_sr.zip
export ZUCO_V2_NR_URL=https://files.de-1.osf.io/v1/resources/2urht/providers/osfstorage/5ddea2baab905e0009e77b0d/?zip=zuco2_nr.zip

export ONESTOP_URL=https://zenodo.org/record/1219041/files/nishkalavallabhi/OneStopEnglishCorpus-bea2018.zip
export ONESTOP_ZIP_PATH=data/readability/OneStopEnglishCorpus-bea2018.zip
export ONESTOP_ADV=nishkalavallabhi-OneStopEnglishCorpus-089be0f/Texts-SeparatedByReadingLevel/Adv-Txt
export ONESTOP_INT=nishkalavallabhi-OneStopEnglishCorpus-089be0f/Texts-SeparatedByReadingLevel/Ele-Txt
export ONESTOP_ELE=nishkalavallabhi-OneStopEnglishCorpus-089be0f/Texts-SeparatedByReadingLevel/Int-Txt
export ONESTOP_RAW_PATH=data/readability/raw_texts

export SST_URL=http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip
export SST_ZIP_PATH=data/eval/stanfordSentimentTreebank.zip
export SST_SENT=stanfordSentimentTreebank/datasetSentences.txt
export SST_NAME=data/eval/sst.tsv

export GP_MVRR="http://syntaxgym.org/test_suite/download?test_suite=265&format=json"
export GP_NPZ_VT="http://syntaxgym.org/test_suite/download?test_suite=253&format=json"
export GP_NPZ_OO="http://syntaxgym.org/test_suite/download?test_suite=274&format=json"

export LOG_FILE=logs/setup.log

GREEN='\033[0;32m'
ORANGE='\033[0;33m'
NC='\033[0m'

echo -e "${GREEN}Creating the project folder structure...${NC}"

mkdir -p data/complexity

mkdir -p models
mkdir -p logs

touch $LOG_FILE

echo -e "${GREEN}Logging outputs to ${LOG_FILE}"
echo -e "${GREEN}Installing lingcomp requirements..."

pip install -r requirements.txt &> $LOG_FILE

echo -e "${GREEN}Installing lingcomp library egg locally..."

make build &>> $LOG_FILE

echo -e "${GREEN}Downloading eye-tracking data..."
wget -P data/eyetracking $GECO 
wget -P data/eyetracking $GECO_EXTRA 

echo -e "${GREEN}Downloading and decompressing readability data..."
wget -P data/readability $ONESTOP_URL 
unzip -o -j $ONESTOP_ZIP_PATH "${ONESTOP_ELE}/"* -d $ONESTOP_RAW_PATH &>> $LOG_FILE
unzip -o -j $ONESTOP_ZIP_PATH "${ONESTOP_INT}/"* -d $ONESTOP_RAW_PATH &>> $LOG_FILE
unzip -o -j $ONESTOP_ZIP_PATH "${ONESTOP_ADV}/"* -d $ONESTOP_RAW_PATH &>> $LOG_FILE
rm -rf $ONESTOP_ZIP_PATH &>> $LOG_FILE

echo -e "${GREEN}Downloading SST data for PWCCA/RSA evaluation..."
wget -P data/eval $SST_URL 
unzip -j $SST_ZIP_PATH $SST_SENT -d "data/eval/" &>> $LOG_FILE
rm -rf $SST_ZIP_PATH &>> $LOG_FILE
mv "data/eval/datasetSentences.txt" $SST_NAME &>> $LOG_FILE
# Column name required by the FARM library
sed -i '1s/.*/sentence_index\ttext/' $SST_NAME &>> $LOG_FILE

echo -e "${GREEN}Downloading Garden Path sentences..."
mkdir -p data/garden_paths
wget -O data/garden_paths/mvrr.json $GP_MVRR
wget -O data/garden_paths/npz_ambig.json $GP_NPZ_VT
wget -O data/garden_paths/npz_obj.json $GP_NPZ_OO

if [ "$DOWNLOAD_ZUCO_MAT_FILES" = true ] ; then
     echo -e "${GREEN}Downloading ZuCO V1/V2 MAT files, this will take some time..."
     wget -O zuco1_nr.zip $ZUCO_V1_NR_URL
     wget -O zuco1_sr.zip $ZUCO_V1_SR_URL
     wget -O zuco2.zip $ZUCO_V2_NR_URL
     unzip -d data/eyetracking/zuco1-nr zuco1_nr.zip
     unzip -d data/eyetracking/zuco1-sr zuco1_sr.zip
     unzip -d data/eyetracking/zuco2 zuco2.zip
     rm -rf zuco*.zip
fi

echo -e "${ORANGE}Complexity data requires additional information for downloading." \
     "Find it here: http://www.italianlp.it/resources/corpus-of-sentences-rated-with-human-complexity-judgments/download-english-sentences/" \
     "Once downloaded, put the file 'complexity_ds_en.csv' in the 'data/complexity' folder${NC}"

echo -e "${GREEN}Done!${NC}"