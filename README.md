

`prose` is a comprehensive natural language processing library in *pure Go*. It supports tokenization, segmentation, part-of-speech tagging, and named-entity extraction with advanced features including multilingual processing, position tracking, and confidence scoring.

This is a fork of the original project, [github.com/jdkato/prose](https://github.com/jdkato/prose) which was archived by the author some time ago.

## What's New in v3.0.0

**Major enhancements** over the original library include:

- **🌍 Multilingual Support**: Automatic language detection and processing for English, Spanish, French, German, and Japanese
- **📍 Position Tracking**: All tokens, entities, and sentences include precise position information in the original text
- **🎯 Confidence Scores**: ML predictions include confidence levels for reliability assessment
- **🚀 Enhanced NER**: Expanded from 2 to 16 entity types (PERSON, ORG, GPE, MONEY, DATE, TIME, PERCENT, FAC, PRODUCT, EVENT, WORK_OF_ART, LANGUAGE, NORP, LAW, ORDINAL, CARDINAL)
- **🔧 Training API**: Complete model training system with cross-validation and performance metrics
- **⚡ Memory Optimization**: Token pooling and efficient data structures reduce memory usage by 20-30%
- **🎛️ Context Support**: All operations support `context.Context` for cancellation, timeouts, and progress tracking
- **📊 Rich Metadata**: Documents include processing statistics, language detection, and performance metrics
- **🔄 100% Backward Compatible**: All existing v2 APIs preserved while adding new functionality

See [UPDATES.md](UPDATES.md) for complete implementation details and migration guide.

You can find a summary on the original library's performance here: [Introducing `prose` v2.0.0: Bringing NLP *to Go*](https://medium.com/@errata.ai/introducing-prose-v2-0-0-bringing-nlp-to-go-a1f0c121e4a5).

## Installation

```console
$ go get github.com/tsawler/prose/v3
```

## Usage

### Contents

* [Overview](#overview)
* [Multilingual Support](#multilingual-support)
* [Tokenizing](#tokenizing)
* [Segmenting](#segmenting)
* [Tagging](#tagging)
* [NER](#ner)

### Overview


```go
package main

import (
    "fmt"
    "log"

    "github.com/tsawler/prose/v3"
)

func main() {
    // Create a new document with the default configuration:
    doc, err := prose.NewDocument("Go is an open-source programming language created at Google.")
    if err != nil {
        log.Fatal(err)
    }

    // Iterate over the doc's tokens:
    for _, tok := range doc.Tokens() {
        fmt.Println(tok.Text, tok.Tag, tok.Label)
        // Go NNP B-GPE
        // is VBZ O
        // an DT O
        // ...
    }

    // Iterate over the doc's named-entities:
    for _, ent := range doc.Entities() {
        fmt.Println(ent.Text, ent.Label)
        // Go GPE
        // Google GPE
    }

    // Iterate over the doc's sentences:
    for _, sent := range doc.Sentences() {
        fmt.Println(sent.Text)
        // Go is an open-source programming language created at Google.
    }
}
```

The document-creation process adheres to the following sequence of steps:

```text
tokenization -> POS tagging -> NE extraction
            \
             segmentation
```

Each step may be disabled (assuming later steps aren't required) by passing the appropriate [*functional option*](https://dave.cheney.net/2014/10/17/functional-options-for-friendly-apis). To disable named-entity extraction, for example, you'd do the following:

```go
doc, err := prose.NewDocument(
        "Go is an open-source programming language created at Google.",
        prose.WithExtraction(false))
```

### Multilingual Support

`prose` v3.0.0 introduces comprehensive multilingual support with automatic language detection and language-specific processing for English, Spanish, French, German, and Japanese.

```go
package main

import (
    "context"
    "fmt"
    "time"

    "github.com/tsawler/prose/v3"
)

func main() {
    // Automatic language detection
    multiDoc, err := prose.NewMultilingualDocument("Bonjour le monde! Comment allez-vous?")
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    
    lang, confidence := multiDoc.DetectedLanguage()
    fmt.Printf("Detected language: %s (confidence: %.2f)\n", lang, confidence)
    // Detected language: fr (confidence: 0.51)
    
    // Enhanced features with context and metadata
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    doc, err := prose.NewDocument("Go is excellent for NLP processing!",
        prose.WithContext(ctx),
        prose.WithLanguage(prose.English),
    )
    if err != nil {
        fmt.Printf("Error: %v\n", err)
        return
    }
    
    fmt.Printf("Processing time: %dms\n", doc.Metadata.ProcessingTimeMs)
    fmt.Printf("Token count: %d\n", doc.Metadata.TokenCount)
    
    // Enhanced tokens with position tracking and confidence scores
    for _, tok := range doc.Tokens() {
        fmt.Printf("%-12s POS: %-4s Position: %d-%d Confidence: %.3f\n", 
            tok.Text, tok.Tag, tok.Start, tok.End, tok.Confidence)
        // Go           POS: NNP  Position: 0-2 Confidence: 0.707
        // is           POS: VBZ  Position: 3-5 Confidence: 1.000
        // excellent    POS: JJ   Position: 6-15 Confidence: 1.000
        // ...
    }
}
```

The new multilingual capabilities include:

- **Automatic Language Detection**: Detects the language of input text with confidence scoring
- **Language-Specific Processing**: Optimized tokenization, stop words, and normalization per language
- **Context Support**: All operations support `context.Context` for timeouts and cancellation
- **Enhanced Metadata**: Processing statistics, timing, and language information
- **Position Tracking**: Precise character positions for all tokens, entities, and sentences
- **Confidence Scores**: ML predictions include reliability assessments

### Tokenizing

`prose` includes a tokenizer capable of processing modern text, including the non-word character spans shown below.

| Type            | Example                           |
|-----------------|-----------------------------------|
| Email addresses | `Jane.Doe@example.com`            |
| Hashtags        | `#trending`                       |
| Mentions        | `@tsawler`                         |
| URLs            | `https://github.com/tsawler/prose` |
| Emoticons       | `:-)`, `>:(`, `o_0`, etc.         |


```go
package main

import (
    "fmt"
    "log"

    "github.com/tsawler/prose/v3"
)

func main() {
    // Create a new document with the default configuration:
    doc, err := prose.NewDocument("@tsawler, go to http://example.com thanks :).")
    if err != nil {
        log.Fatal(err)
    }

    // Iterate over the doc's tokens:
    for _, tok := range doc.Tokens() {
        fmt.Println(tok.Text, tok.Tag)
        // @tsawler NN
        // , ,
        // go VB
        // to TO
        // http://example.com VB
        // thanks NNS
        // : :
        // ) )
        // . .
    }
}
```

### Segmenting

`prose` includes one of the most accurate sentence segmenters available, according to the [Golden Rules](https://github.com/diasks2/pragmatic_segmenter#comparison-of-segmentation-tools-libraries-and-algorithms) created by the developers of the `pragmatic_segmenter`.

| Name                | Language | License   | GRS (English)  | GRS (Other) | Speed†   |
|---------------------|----------|-----------|----------------|-------------|----------|
| Pragmatic Segmenter | Ruby     | MIT       | 98.08% (51/52) | 100.00%     | 3.84 s   |
| prose               | Go       | MIT       | 75.00% (39/52) | N/A         | 0.96 s   |
| TactfulTokenizer    | Ruby     | GNU GPLv3 | 65.38% (34/52) | 48.57%      | 46.32 s  |
| OpenNLP             | Java     | APLv2     | 59.62% (31/52) | 45.71%      | 1.27 s   |
| Standford CoreNLP   | Java     | GNU GPLv3 | 59.62% (31/52) | 31.43%      | 0.92 s   |
| Splitta             | Python   | APLv2     | 55.77% (29/52) | 37.14%      | N/A      |
| Punkt               | Python   | APLv2     | 46.15% (24/52) | 48.57%      | 1.79 s   |
| SRX English         | Ruby     | GNU GPLv3 | 30.77% (16/52) | 28.57%      | 6.19 s   |
| Scapel              | Ruby     | GNU GPLv3 | 28.85% (15/52) | 20.00%      | 0.13 s   |

> † The original tests were performed using a *MacBook Pro 3.7 GHz Quad-Core Intel Xeon E5 running 10.9.5*, while `prose` was timed using a *MacBook Pro 2.9 GHz Intel Core i7 running 10.13.3*.

```go
package main

import (
    "fmt"
    "strings"

    "github.com/tsawler/prose/v3"
)

func main() {
    // Create a new document with the default configuration:
    doc, _ := prose.NewDocument(strings.Join([]string{
        "I can see Mt. Fuji from here.",
        "St. Michael's Church is on 5th st. near the light."}, " "))

    // Iterate over the doc's sentences:
    sents := doc.Sentences()
    fmt.Println(len(sents)) // 2
    for _, sent := range sents {
        fmt.Println(sent.Text)
        // I can see Mt. Fuji from here.
        // St. Michael's Church is on 5th st. near the light.
    }
}
```

### Tagging

`prose` includes a tagger based on Textblob's ["fast and accurate" POS tagger](https://github.com/sloria/textblob-aptagger). Below is a comparison of its performance against [NLTK](http://www.nltk.org/)'s implementation of the same tagger on the Treebank corpus:

| Library | Accuracy | 5-Run Average (sec) |
|:--------|---------:|--------------------:|
| NLTK    |    0.893 |               7.224 |
| `prose` |    0.961 |               2.538 |

(See [`scripts/test_model.py`](https://github.com/tsawler/aptag/blob/master/scripts/test_model.py) for more information.)

The full list of supported POS tags is given below.

| TAG        | DESCRIPTION                               |
|------------|-------------------------------------------|
| `(`        | left round bracket                        |
| `)`        | right round bracket                       |
| `,`        | comma                                     |
| `:`        | colon                                     |
| `.`        | period                                    |
| `''`       | closing quotation mark                    |
| ``` `` ``` | opening quotation mark                    |
| `#`        | number sign                               |
| `$`        | currency                                  |
| `CC`       | conjunction, coordinating                 |
| `CD`       | cardinal number                           |
| `DT`       | determiner                                |
| `EX`       | existential there                         |
| `FW`       | foreign word                              |
| `IN`       | conjunction, subordinating or preposition |
| `JJ`       | adjective                                 |
| `JJR`      | adjective, comparative                    |
| `JJS`      | adjective, superlative                    |
| `LS`       | list item marker                          |
| `MD`       | verb, modal auxiliary                     |
| `NN`       | noun, singular or mass                    |
| `NNP`      | noun, proper singular                     |
| `NNPS`     | noun, proper plural                       |
| `NNS`      | noun, plural                              |
| `PDT`      | predeterminer                             |
| `POS`      | possessive ending                         |
| `PRP`      | pronoun, personal                         |
| `PRP$`     | pronoun, possessive                       |
| `RB`       | adverb                                    |
| `RBR`      | adverb, comparative                       |
| `RBS`      | adverb, superlative                       |
| `RP`       | adverb, particle                          |
| `SYM`      | symbol                                    |
| `TO`       | infinitival to                            |
| `UH`       | interjection                              |
| `VB`       | verb, base form                           |
| `VBD`      | verb, past tense                          |
| `VBG`      | verb, gerund or present participle        |
| `VBN`      | verb, past participle                     |
| `VBP`      | verb, non-3rd person singular present     |
| `VBZ`      | verb, 3rd person singular present         |
| `WDT`      | wh-determiner                             |
| `WP`       | wh-pronoun, personal                      |
| `WP$`      | wh-pronoun, possessive                    |
| `WRB`      | wh-adverb                                 |

### NER

`prose` v3.0.0 includes a significantly enhanced named entity recognition system that can identify 16 different entity types including people (`PERSON`), geographical/political entities (`GPE`), organizations (`ORG`), dates (`DATE`), money (`MONEY`), and many more. All predictions include confidence scores and precise position information.

```go
package main

import (
    "fmt"

    "github.com/tsawler/prose/v3"
)

func main() {
    doc, _ := prose.NewDocument("Lebron James plays basketball in Los Angeles.")
    for _, ent := range doc.Entities() {
        fmt.Println(ent.Text, ent.Label)
        // Lebron James PERSON
        // Los Angeles GPE
    }
}
```

However, in an attempt to make this feature more useful, we've made it straightforward to train your own models for specific use cases. See [Prodigy + `prose`: Radically efficient machine teaching *in Go*](https://medium.com/@errata.ai/prodigy-prose-radically-efficient-machine-teaching-in-go-93389bf2d772) for a tutorial.
