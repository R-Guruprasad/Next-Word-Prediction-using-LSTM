# Next-Word-Prediction-using-LSTM

## AIM

To develop an LSTM-based model for predicting the next word in a text corpus.

## Problem Statement and Dataset
Build a neural network-based text generator to produce new sequences from a given corpus by tokenizing text, creating n-grams, padding sequences, and encoding labels.

![image](https://github.com/user-attachments/assets/85aebfdd-024e-40cf-af1f-e0b1da221079)

## DESIGN STEPS

### STEP 1:
Use fit_vectorizer to initialize and fit a TextVectorization layer on the corpus for word-to-integer tokenization.
### STEP 2:
Generate n-grams for each sentence using n_gram_seqs, creating sequential input data.
### STEP 3:
Pad these sequences to a uniform length with pad_seqs, enabling consistent input shapes for training.
### STEP 4:
Split each sequence into features and labels, where features contain all words except the last, and labels are the last word.
### STEP 5:
One-hot encode the labels with a vocabulary size from total_words for categorical prediction.
### STEP 6:
Construct a TensorFlow dataset with these features and labels, batching them for efficient processing.
### STEP 7:
Build the model with an Embedding layer, Bidirectional LSTM for sequence processing, and Dense layer with softmax for word prediction.
### STEP 8:
Compile and train the model using categorical cross-entropy loss and the Adam optimizer.

## PROGRAM
### Name: R Guruprasad
### Register Number: 212222240033

### 1.fit_vectorizer function

```py
def fit_vectorizer(corpus):
    """
    Instantiates the vectorizer class on the corpus

    Args:
        corpus (list): List with the sentences.

    Returns:
        (tf.keras.layers.TextVectorization): an instance of the TextVectorization class containing the word-index dictionary,
        adapted to the corpus sentences.
    """

    tf.keras.utils.set_random_seed(65) # Do not change this line or you may have different expected outputs throughout the
    assignment

    ### START CODE HERE ###

     # Define the object with appropriate parameters
    vectorizer =tf.keras.layers.TextVectorization(
        standardize= 'lower_and_strip_punctuation', # Convert to lowercase and strip punctuation
        split='whitespace',               # Split on whitespace (default)
        ragged=True,             # Allow ragged tensors
        output_mode='int'               # Output as integers
    )

    # Adapt it to the corpus
    vectorizer.adapt(corpus)
    ### END CODE HERE ###

    return vectorizer
```

### 2. n_grams_seqs function

```py
def n_gram_seqs(corpus, vectorizer):
    """
    Generates a list of n-gram sequences

    Args:
        corpus (list of string): lines of texts to generate n-grams for
        vectorizer (tf.keras.layers.TextVectorization): an instance of the TextVectorization class adapted in the corpus

    Returns:
        (list of tf.int64 tensors): the n-gram sequences for each line in the corpus
    """
    input_sequences = []

    ### START CODE HERE ###
    for sentence in corpus:
        # Vectorize the sentence to get the token indices
        vectorized_sentence =vectorizer(sentence)

        # Generate n-grams for the vectorized sentence
        for i in range(2,len(vectorized_sentence)+1):  # Start from 2 to avoid the first token
            n_gram =vectorized_sentence[:i]
            input_sequences.append(n_gram)

    ### END CODE HERE ###

    return input_sequences
```

### 3. pad_seqs function

```py
def pad_seqs(input_sequences, max_sequence_len):
    """
    Pads tokenized sequences to the same length

    Args:
        input_sequences (list of int): tokenized sequences to pad
        maxlen (int): maximum length of the token sequences

    Returns:
        (np.array of int32): tokenized sequences padded to the same length
    """

   ### START CODE HERE ###
    # Convert tensors to lists if necessary
    input_list = [seq if isinstance(seq, list) else seq.numpy().tolist() for seq in input_sequences]

    # Use pad_sequences to pad the sequences with left padding ('pre')
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        input_sequences,
        maxlen=max_sequence_len,
        padding='pre',
        dtype='int32'
                     # Use the list of lists for padding
                     # Set the maximum length
                     # Pad sequences to the left (before the sequence)
                     # Specify the output type as int32
    )
    ### END CODE HERE ###

    return padded_sequences
```

### 4. features_and_labels_dataset function

```py
def features_and_labels_dataset(input_sequences, total_words):
    """
    Generates features and labels from n-grams and returns a tensorflow dataset

    Args:
        input_sequences (list of int): sequences to split features and labels from
        total_words (int): vocabulary size

    Returns:
        (tf.data.Dataset): Dataset with elements in the form (sentence, label)
    """
    ### START CODE HERE ###
    # Define the features by taking all tokens except the last one for each sequence
    features =input_sequences[:,:-1]

    # Define the labels by taking the last token for each sequence
    labels =input_sequences[:,-1] 

    # One-hot encode the labels using total_words as the number of classes
    one_hot_labels =tf.keras.utils.to_categorical(labels, num_classes=total_words)

    # Build the dataset using the features and one-hot encoded labels
    dataset = tf.data.Dataset.from_tensor_slices((features, one_hot_labels))

    # Batch the dataset with a batch size of 16
    batch_size = 16  # Feel free to adjust this based on the global variable, but should be <= 64
    batched_dataset = dataset.batch(batch_size)

    ### END CODE HERE ###

    return batched_dataset
```

### 5.create_model function

```py
def create_model(total_words, max_sequence_len):
    """
    Creates a text generator model

    Args:
        total_words (int): size of the vocabulary for the Embedding layer input
        max_sequence_len (int): length of the input sequences

    Returns:
       (tf.keras Model): the text generator model
    """
    model = tf.keras.Sequential()

   ### START CODE HERE ###
    # Input layer shape is max_sequence_len - 1 because we removed the last word as a label
    model.add(tf.keras.layers.Input(shape=(max_sequence_len - 1,)))

    # Embedding layer
    model.add(tf.keras.layers.Embedding(input_dim=total_words, output_dim=100, input_length=max_sequence_len - 1))

    # Add a Bidirectional LSTM layer with 150 units
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128)))

    # Add a Dense layer with 'total_words' units and softmax activation
    model.add(tf.keras.layers.Dense(total_words, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',  # Loss function for multi-class classification
                  optimizer='adam',                 # Adam optimizer
                  metrics=['accuracy'])
    ### END CODE HERE ###

    return model
```



## OUTPUT
### 1. fit_vectorizer output

![image](https://github.com/user-attachments/assets/76ce1dca-3b20-4e1a-a641-8326707787ab)

### 2. n_grams_seqs output

![image](https://github.com/user-attachments/assets/cf5f91e1-1ccb-40d6-a444-244d074ef428)

### 3. pad_seqs output

![image](https://github.com/user-attachments/assets/cd83595e-d31b-416a-9261-c14b283edaec)

### 4. features_and_labels_dataset output

![image](https://github.com/user-attachments/assets/955913d6-51b8-46cc-ad3d-e5fafeb5e3ee)

### 5. Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/9e6e56a9-5f46-4f97-ad3e-ce0df6248717)

### 6. Sample Text Prediction

![image](https://github.com/user-attachments/assets/e8ba111b-9327-4faa-b03d-9d477231d953)

## RESULT

Thus, a trained text generator model capable of predicting the next word in a sequence from the given corpus is successfully implelemted.


