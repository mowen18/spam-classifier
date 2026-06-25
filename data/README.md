# Data

This project uses the UCI SMS Spam Collection.

- Dataset name: SMS Spam Collection
- UCI dataset ID: 228
- DOI: 10.24432/C5CC84
- License: CC BY 4.0
- Creators: Tiago Almeida and José María Gómez Hidalgo

The raw dataset file should be stored at:

```text
data/raw/SMSSpamCollection
```

If the file is not present, retrieve it with:

```bash
python -c "from spam_classifier.data import download_sms_spam_collection; download_sms_spam_collection()"
```

The retrieval helper first attempts `ucimlrepo` with dataset ID 228. If UCI reports that the dataset is listed but unavailable for direct package import, the helper falls back to UCI's official static archive for the same dataset ID. The modeling workflow does not alter the raw file.
