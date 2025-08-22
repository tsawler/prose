# External Lexicon Guide

This guide explains how to extend Prose's sentiment analysis capabilities using external JSON lexicon files. The hybrid approach allows you to use Prose's built-in core sentiment words while adding your own custom words and domain-specific vocabulary.

## Overview

Prose sentiment analysis supports a **hybrid lexicon approach**:

1. **Core lexicons** - Essential sentiment words hardcoded in the library for reliability
2. **External lexicons** - Custom word lists loaded from JSON files for extensibility

External lexicons are merged with core lexicons, allowing you to:
- Add domain-specific sentiment words (e.g., financial, medical, social media)
- Customize sentiment values for specific use cases
- Support additional languages or dialects
- Update lexicons without recompiling the library

## Supported Languages

The external lexicon system supports all languages available in Prose:

- `english` - English language sentiment words
- `spanish` - Spanish language sentiment words  
- `french` - French language sentiment words
- `german` - German language sentiment words
- `japanese` - Japanese language sentiment words

You can specify one or more languages in a single JSON file.

## JSON Format

### Basic Structure

```json
{
  "languages": {
    "english": {
      "words": [...],
      "positive": [...],
      "negative": [...],
      "modifiers": [...],
      "intensifiers": [...],
      "diminishers": [...],
      "negations": [...]
    },
    "spanish": {
      // Spanish word lists...
    }
  }
}
```

### Word Categories

#### 1. General Sentiment Words (`words`)

For words with explicit sentiment scores:

```json
"words": [
  {
    "word": "fantastic",
    "sentiment": 0.8,
    "confidence": 0.9,
    "pos": ["JJ"],
    "domain": "general"
  },
  {
    "word": "disappointing",
    "sentiment": -0.6,
    "confidence": 0.85,
    "pos": ["VBG", "JJ"],
    "domain": "general"
  }
]
```

**Fields:**
- `word` (required): The sentiment word
- `sentiment` (required): Sentiment score from -1.0 (very negative) to 1.0 (very positive)
- `confidence` (required): Confidence in the sentiment score (0.0 to 1.0)
- `pos` (optional): Part-of-speech tags where this sentiment applies
- `domain` (optional): Domain specificity (e.g., "financial", "medical", "social")

#### 2. Positive Words (`positive`)

Simplified format for positive sentiment words:

```json
"positive": [
  {
    "word": "excellent",
    "sentiment": 0.85,
    "confidence": 0.9
  },
  {
    "word": "outstanding",
    "sentiment": 0.9,
    "confidence": 0.95
  }
]
```

#### 3. Negative Words (`negative`)

Simplified format for negative sentiment words:

```json
"negative": [
  {
    "word": "awful",
    "sentiment": -0.8,
    "confidence": 0.9
  },
  {
    "word": "disappointing", 
    "sentiment": -0.6,
    "confidence": 0.85
  }
]
```

#### 4. Modifiers (`modifiers`)

Words that modify sentiment intensity with custom factors:

```json
"modifiers": [
  {
    "word": "extremely",
    "factor": 2.0
  },
  {
    "word": "barely",
    "factor": 0.3
  }
]
```

#### 5. Intensifiers (`intensifiers`)

Words that strengthen sentiment (uses default factor of 1.5):

```json
"intensifiers": [
  "incredibly",
  "tremendously",
  "exceptionally",
  "remarkably"
]
```

#### 6. Diminishers (`diminishers`)

Words that weaken sentiment (uses default factor of 0.5):

```json
"diminishers": [
  "marginally",
  "moderately", 
  "minimally",
  "nominally"
]
```

#### 7. Negations (`negations`)

Words that negate or reverse sentiment:

```json
"negations": [
  "not",
  "never",
  "without",
  "lacks"
]
```

## Complete Example

Here's a complete example for financial domain sentiment analysis:

```json
{
  "languages": {
    "english": {
      "positive": [
        {
          "word": "profitable",
          "sentiment": 0.7,
          "confidence": 0.9,
          "domain": "financial"
        },
        {
          "word": "bullish",
          "sentiment": 0.6,
          "confidence": 0.85,
          "domain": "financial"
        },
        {
          "word": "outperform",
          "sentiment": 0.65,
          "confidence": 0.8,
          "domain": "financial"
        }
      ],
      "negative": [
        {
          "word": "bearish",
          "sentiment": -0.6,
          "confidence": 0.85,
          "domain": "financial"
        },
        {
          "word": "underperform",
          "sentiment": -0.65,
          "confidence": 0.8,
          "domain": "financial"
        },
        {
          "word": "volatile",
          "sentiment": -0.4,
          "confidence": 0.7,
          "domain": "financial"
        }
      ],
      "intensifiers": [
        "significantly",
        "substantially",
        "dramatically"
      ],
      "diminishers": [
        "moderately",
        "slightly"
      ],
      "modifiers": [
        {
          "word": "massively",
          "factor": 2.5
        }
      ]
    },
    "spanish": {
      "positive": [
        {
          "word": "rentable",
          "sentiment": 0.7,
          "confidence": 0.9,
          "domain": "financial"
        }
      ],
      "negative": [
        {
          "word": "volátil",
          "sentiment": -0.4,
          "confidence": 0.7,
          "domain": "financial"
        }
      ]
    }
  }
}
```

## Usage Examples

### 1. Basic Usage with External Lexicon

```go
package main

import (
    "fmt"
    "log"
    "github.com/tsawler/prose/v3"
)

func main() {
    // Create analyzer with external lexicon
    config := prose.DefaultSentimentConfig()
    analyzer, err := prose.NewSentimentAnalyzerWithExternal(
        prose.English, 
        config, 
        "financial_sentiment.json",
    )
    if err != nil {
        log.Fatal("Error loading external lexicon:", err)
    }

    // Analyze financial text
    text := "The stock shows bullish momentum and significantly outperformed expectations"
    doc, _ := prose.NewDocument(text)
    sentiment := analyzer.AnalyzeDocument(doc)
    
    fmt.Printf("Polarity: %.2f, Intensity: %.2f, Confidence: %.2f\n", 
        sentiment.Polarity, sentiment.Intensity, sentiment.Confidence)
}
```

### 2. Loading Multiple Languages

```go
// Load lexicon for multiple languages
lexicon := prose.LoadSentimentLexicon(prose.English)
err := lexicon.LoadExternalLexicon("multilingual_sentiment.json", 
    []prose.Language{prose.English, prose.Spanish, prose.French})
if err != nil {
    log.Fatal("Error loading multilingual lexicon:", err)
}
```

### 3. Runtime Lexicon Updates

```go
// Load base analyzer
analyzer := prose.NewSentimentAnalyzer(prose.English, config)

// Later, add domain-specific words
err := analyzer.lexicon.LoadExternalLexicon("medical_sentiment.json", 
    []prose.Language{prose.English})
if err != nil {
    log.Printf("Warning: Could not load medical lexicon: %v", err)
}
```

## Best Practices

### 1. Lexicon Design

- **Start small**: Begin with 20-50 high-confidence domain words
- **Balanced vocabulary**: Include both positive and negative domain terms
- **Appropriate confidence**: Use higher confidence (0.8+) for clearly sentiment words, lower (0.6-0.7) for ambiguous terms
- **POS tags**: Specify part-of-speech when sentiment depends on word usage

### 2. Sentiment Scoring

- **Conservative scores**: Avoid extreme values (-1.0, 1.0) unless absolutely certain
- **Domain calibration**: Test sentiment scores against domain expert judgment
- **Iterative refinement**: Start with approximate scores and refine based on testing

### 3. File Organization

- **Domain separation**: Create separate files for different domains
- **Language grouping**: Group related languages in single files when they share vocabulary
- **Version control**: Track lexicon changes with semantic versioning

### 4. Performance Considerations

- **File size**: Keep individual JSON files under 1MB for fast loading
- **Caching**: External lexicons are loaded once and cached in memory
- **Selective loading**: Only load languages you actually need

## Error Handling

The external lexicon system provides detailed error messages:

```go
analyzer, err := prose.NewSentimentAnalyzerWithExternal(
    prose.English, config, "sentiment.json")
if err != nil {
    switch {
    case strings.Contains(err.Error(), "no such file"):
        log.Println("Lexicon file not found, using core lexicon only")
        analyzer = prose.NewSentimentAnalyzer(prose.English, config)
    case strings.Contains(err.Error(), "parsing"):
        log.Fatal("Invalid JSON format in lexicon file")
    default:
        log.Fatal("Unexpected error:", err)
    }
}
```

## Validation

The system validates external lexicon files:

- **JSON structure**: Must match the documented schema
- **Sentiment ranges**: Sentiment scores must be between -1.0 and 1.0
- **Confidence ranges**: Confidence scores must be between 0.0 and 1.0
- **Required fields**: `word`, `sentiment`, and `confidence` are required for word entries

## Integration with Feature Extraction

External lexicons integrate with feature extraction for ML models:

- Words from external lexicons are included in semantic features
- Custom modifiers affect feature importance weighting
- Domain-specific words create specialized feature dimensions

## Troubleshooting

### Common Issues

1. **File not found**: Ensure the JSON file path is correct and accessible
2. **JSON parsing errors**: Validate JSON syntax using a JSON validator
3. **Silent failures**: Check that language keys match exactly ("english", not "English")
4. **Missing words**: Verify that words are properly lowercase in the JSON file
5. **Unexpected sentiment**: Check for typos in word spelling

### Debug Tips

```go
// Enable verbose logging to see lexicon loading details
import "log"

log.SetFlags(log.LstdFlags | log.Lshortfile)
log.Println("Loading external lexicon...")

analyzer, err := prose.NewSentimentAnalyzerWithExternal(...)
if err != nil {
    log.Printf("Lexicon error: %v", err)
}
```

## Migration from Hardcoded Words

If you're currently modifying the source code to add custom words:

1. **Extract your words**: Copy custom words from source code to JSON format
2. **Test compatibility**: Verify sentiment scores produce similar results  
3. **Deploy gradually**: Start with a subset of custom words
4. **Remove hardcoded changes**: Clean up source modifications after validation

This hybrid approach provides the reliability of core lexicons with the flexibility of external customization, making Prose sentiment analysis adaptable to diverse domains and use cases.