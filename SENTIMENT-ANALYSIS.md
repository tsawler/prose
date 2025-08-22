# Sentiment Analysis Implementation Guide for Prose NLP

## Implementation Status ✅

**Current Status: Phase 1 Complete + Hybrid Lexicon Enhancement**

- ✅ **Core Implementation**: Full sentiment analysis with polarity, intensity, and confidence scoring
- ✅ **Multilingual Support**: Complete support for English, Spanish, French, German, and Japanese
- ✅ **Hybrid Lexicon System**: Core hardcoded lexicons + external JSON file support
- ✅ **Advanced Features**: Negation handling, modifier effects, feature extraction for ML
- ✅ **Comprehensive Testing**: Full test suite with multilingual and external lexicon validation
- 🔄 **Pending**: ML model integration (Phase 2), aspect-based analysis (Phase 3)

**Recent Enhancement**: Added hybrid lexicon approach allowing users to extend sentiment analysis with custom JSON lexicons for domain-specific applications (financial, medical, social media, etc.). See [EXTERNAL-LEXICON-GUIDE.md](./EXTERNAL-LEXICON-GUIDE.md) for usage details.

## Table of Contents
1. [Overview](#overview)
2. [Architecture Design](#architecture-design)
3. [Type Definitions](#type-definitions)
4. [Core Components](#core-components)
5. [Implementation Details](#implementation-details)
6. [API Design](#api-design)
7. [Testing Strategy](#testing-strategy)
8. [Performance Considerations](#performance-considerations)
9. [Integration Checklist](#integration-checklist)

## Overview

This document provides a comprehensive implementation guide for adding sentiment analysis capabilities to the Prose NLP library. The feature will support:
- **Polarity Analysis**: Determining positive, negative, or neutral sentiment (-1.0 to 1.0 scale)
- **Intensity Analysis**: Measuring the strength of sentiment (0.0 to 1.0 scale)
- **Confidence Scoring**: Providing reliability metrics for predictions (0.0 to 1.0 scale)
- **Multi-level Analysis**: Document, sentence, and aspect-level sentiment
- **Multilingual Support**: Complete support for English, Spanish, French, German, and Japanese
- **Hybrid Lexicon System**: Core reliability + external JSON customization for domain-specific applications

## Hybrid Lexicon System ✅

The implemented sentiment analysis uses a **hybrid approach** combining:

1. **Core Lexicons** (Hardcoded): Essential sentiment words built into the library for reliability
2. **External Lexicons** (JSON): Custom domain-specific words loaded from user-provided files

**Benefits:**
- ✅ **Reliability**: Core words always available, no external dependencies for basic functionality
- ✅ **Extensibility**: Add domain-specific vocabulary (financial, medical, social media) without recompilation
- ✅ **Flexibility**: Update sentiment lexicons without modifying source code
- ✅ **Performance**: External lexicons loaded once and cached in memory

**Usage Example:**
```go
// Standard analyzer with core lexicons only
analyzer := NewSentimentAnalyzer(English, DefaultSentimentConfig())

// Enhanced analyzer with custom financial lexicon
analyzer, err := NewSentimentAnalyzerWithExternal(English, config, "financial_lexicon.json")
```

See [EXTERNAL-LEXICON-GUIDE.md](./EXTERNAL-LEXICON-GUIDE.md) for complete usage documentation.

## Architecture Design

### File Structure
```
prose/
├── sentiment.go              # Core sentiment analysis implementation ✅
├── sentiment_lexicon.go      # Lexicon management and loading ✅
├── sentiment_features.go     # Feature extraction for ML models ✅
├── sentiment_test.go         # Unit tests ✅
├── EXTERNAL-LEXICON-GUIDE.md # External lexicon usage guide ✅
├── data/ (external lexicons)
│   └── sentiment/
│       ├── financial_lexicon.json    # Financial domain lexicon (example)
│       ├── medical_lexicon.json      # Medical domain lexicon (example)
│       └── model/ (future Phase 2)
│           ├── weights.gob    # Trained model weights
│           └── features.gob   # Feature mappings
└── testdata/
    └── sentiment/
        ├── test_cases.json
        └── benchmarks.json
```

**Key Changes:**
- ✅ Core files implemented with hybrid lexicon support
- ✅ External lexicon documentation added
- 🔄 External lexicons are user-provided JSON files (not bundled)
- 🔄 ML model files pending Phase 2 implementation

### Component Interaction
```
Document → Tokenizer → POS Tagger → Sentiment Analyzer → Results
                                          ↓
                                    Lexicon Matcher
                                          ↓
                                    Feature Extractor
                                          ↓
                                    ML Classifier
                                          ↓
                                    Score Combiner
```

## Type Definitions

### Core Types (types.go additions)

```go
// SentimentScore represents the sentiment analysis results
type SentimentScore struct {
    // Primary sentiment metrics
    Polarity    float64                    // -1.0 (negative) to 1.0 (positive)
    Intensity   float64                    // 0.0 (neutral) to 1.0 (strong)
    Confidence  float64                    // 0.0 to 1.0 reliability score
    
    // Detailed classification
    Dominant    SentimentClass             // Primary sentiment class
    Scores      map[SentimentClass]float64 // Individual class probabilities
    
    // Subjectivity analysis
    Subjectivity float64                   // 0.0 (objective) to 1.0 (subjective)
    
    // Contributing factors
    Features    SentimentFeatures          // Key features affecting sentiment
}

// SentimentClass represents sentiment categories
type SentimentClass string

const (
    StrongPositive  SentimentClass = "strong_positive"
    Positive        SentimentClass = "positive"
    Neutral         SentimentClass = "neutral"
    Negative        SentimentClass = "negative"
    StrongNegative  SentimentClass = "strong_negative"
    Mixed           SentimentClass = "mixed"  // For conflicting sentiments
)

// SentimentFeatures tracks contributing factors
type SentimentFeatures struct {
    PositiveWords   []WordContribution
    NegativeWords   []WordContribution
    Modifiers       []ModifierEffect
    Negations       []NegationScope
    Intensifiers    []IntensifierEffect
}

// WordContribution represents a word's sentiment contribution
type WordContribution struct {
    Word        string
    Position    int
    BaseScore   float64
    AdjustedScore float64
    Confidence  float64
}

// ModifierEffect represents sentiment modifiers
type ModifierEffect struct {
    Modifier    string
    Type        ModifierType
    Position    int
    Impact      float64
}

// ModifierType categorizes sentiment modifiers
type ModifierType string

const (
    Intensifier ModifierType = "intensifier"  // "very", "extremely"
    Diminisher  ModifierType = "diminisher"   // "slightly", "somewhat"
    Negation    ModifierType = "negation"     // "not", "never"
)

// AspectSentiment for aspect-based sentiment analysis
type AspectSentiment struct {
    Aspect      string
    Text        string
    Sentiment   SentimentScore
    Start       int
    End         int
}

// SentimentContext provides analysis context
type SentimentContext struct {
    Domain      string  // "general", "financial", "medical", etc.
    Sarcasm     bool    // Sarcasm detection flag
    Irony       bool    // Irony detection flag
    Emoji       bool    // Consider emoji sentiment
}
```

## Core Components

### 1. Sentiment Analyzer (sentiment.go)

```go
package prose

import (
    "context"
    "math"
    "strings"
)

// SentimentAnalyzer performs sentiment analysis
type SentimentAnalyzer struct {
    lexicon     *SentimentLexicon
    classifier  *sentimentClassifier
    config      SentimentConfig
}

// SentimentConfig configures sentiment analysis
type SentimentConfig struct {
    UseLexicon      bool
    UseML           bool
    UseContext      bool
    MinConfidence   float64
    NegationWindow  int  // Words to check for negation
    AspectAnalysis  bool
}

// DefaultSentimentConfig returns standard configuration
func DefaultSentimentConfig() SentimentConfig {
    return SentimentConfig{
        UseLexicon:     true,
        UseML:          true,
        UseContext:     true,
        MinConfidence:  0.5,
        NegationWindow: 3,
        AspectAnalysis: false,
    }
}

// NewSentimentAnalyzer creates a sentiment analyzer
func NewSentimentAnalyzer(lang Language, config SentimentConfig) *SentimentAnalyzer {
    return &SentimentAnalyzer{
        lexicon:    LoadSentimentLexicon(lang),
        classifier: loadSentimentModel(lang),
        config:     config,
    }
}

// AnalyzeDocument performs document-level sentiment analysis
func (sa *SentimentAnalyzer) AnalyzeDocument(doc *Document) SentimentScore {
    // Implementation detail below
    sentences := doc.Sentences()
    tokens := doc.Tokens()
    
    // Collect sentence-level sentiments
    sentenceScores := make([]SentimentScore, len(sentences))
    for i, sent := range sentences {
        sentenceScores[i] = sa.AnalyzeSentence(sent, tokens)
    }
    
    // Aggregate to document level
    return sa.aggregateSentiments(sentenceScores)
}

// AnalyzeSentence performs sentence-level analysis
func (sa *SentimentAnalyzer) AnalyzeSentence(sent Sentence, tokens []Token) SentimentScore {
    // Extract relevant tokens for this sentence
    sentTokens := extractSentenceTokens(sent, tokens)
    
    var score SentimentScore
    
    // Step 1: Lexicon-based analysis
    if sa.config.UseLexicon {
        lexiconScore := sa.analyzeLexicon(sentTokens)
        score = combineScores(score, lexiconScore, 0.4)
    }
    
    // Step 2: ML-based analysis
    if sa.config.UseML {
        features := sa.extractFeatures(sentTokens)
        mlScore := sa.classifier.predict(features)
        score = combineScores(score, mlScore, 0.6)
    }
    
    // Step 3: Apply contextual adjustments
    if sa.config.UseContext {
        score = sa.applyContextualRules(score, sentTokens)
    }
    
    return score
}

// analyzeLexicon performs lexicon-based sentiment analysis
func (sa *SentimentAnalyzer) analyzeLexicon(tokens []*Token) SentimentScore {
    var (
        posScore    float64
        negScore    float64
        wordCount   int
        features    SentimentFeatures
    )
    
    for i, token := range tokens {
        // Skip non-content words
        if !isContentWord(token) {
            continue
        }
        
        // Check for negation in preceding context
        negated := sa.checkNegation(tokens, i)
        
        // Get base sentiment score
        sentiment := sa.lexicon.GetSentiment(token.Text)
        if sentiment == 0 {
            sentiment = sa.lexicon.GetSentiment(strings.ToLower(token.Text))
        }
        
        // Apply modifiers
        modified := sa.applyModifiers(sentiment, tokens, i)
        
        // Apply negation
        if negated {
            modified = -modified * 0.5 // Negation reverses but weakens
            features.Negations = append(features.Negations, NegationScope{
                Position: i,
                Scope:    sa.config.NegationWindow,
            })
        }
        
        // Track contribution
        if modified != 0 {
            contrib := WordContribution{
                Word:          token.Text,
                Position:      token.Start,
                BaseScore:     sentiment,
                AdjustedScore: modified,
                Confidence:    sa.lexicon.GetConfidence(token.Text),
            }
            
            if modified > 0 {
                posScore += modified
                features.PositiveWords = append(features.PositiveWords, contrib)
            } else {
                negScore += math.Abs(modified)
                features.NegativeWords = append(features.NegativeWords, contrib)
            }
            wordCount++
        }
    }
    
    // Calculate final scores
    if wordCount == 0 {
        return SentimentScore{
            Polarity:   0,
            Intensity:  0,
            Confidence: 0,
            Dominant:   Neutral,
        }
    }
    
    // Normalize scores
    posScore = posScore / float64(wordCount)
    negScore = negScore / float64(wordCount)
    
    // Calculate polarity (-1 to 1)
    polarity := (posScore - negScore) / (posScore + negScore + 0.00001)
    
    // Calculate intensity (0 to 1)
    intensity := math.Min(1.0, (posScore+negScore)/2)
    
    // Determine dominant class
    dominant := classifyPolarity(polarity, intensity)
    
    // Calculate confidence based on lexicon coverage
    coverage := float64(wordCount) / float64(len(tokens))
    confidence := math.Min(1.0, coverage*2) * 0.7 // Max 0.7 for lexicon-only
    
    return SentimentScore{
        Polarity:   polarity,
        Intensity:  intensity,
        Confidence: confidence,
        Dominant:   dominant,
        Features:   features,
        Scores: map[SentimentClass]float64{
            StrongPositive: calculateClassProb(polarity, intensity, StrongPositive),
            Positive:       calculateClassProb(polarity, intensity, Positive),
            Neutral:        calculateClassProb(polarity, intensity, Neutral),
            Negative:       calculateClassProb(polarity, intensity, Negative),
            StrongNegative: calculateClassProb(polarity, intensity, StrongNegative),
        },
    }
}

// checkNegation detects negation in context
func (sa *SentimentAnalyzer) checkNegation(tokens []*Token, position int) bool {
    start := maxInt(0, position-sa.config.NegationWindow)
    
    for i := start; i < position; i++ {
        if isNegation(tokens[i].Text) {
            // Check if there's a clause boundary between negation and target
            for j := i + 1; j < position; j++ {
                if isClauseBoundary(tokens[j]) {
                    return false
                }
            }
            return true
        }
    }
    return false
}

// applyModifiers adjusts sentiment based on intensifiers/diminishers
func (sa *SentimentAnalyzer) applyModifiers(baseSentiment float64, tokens []*Token, position int) float64 {
    if position == 0 {
        return baseSentiment
    }
    
    prevToken := tokens[position-1]
    modifier := getModifierStrength(prevToken.Text)
    
    if modifier != 0 {
        // Intensifiers multiply sentiment
        return baseSentiment * (1 + modifier)
    }
    
    return baseSentiment
}
```

### 2. Sentiment Lexicon (sentiment_lexicon.go)

```go
package prose

import (
    "encoding/json"
    "sync"
)

// SentimentLexicon manages sentiment word lists
type SentimentLexicon struct {
    words       map[string]LexiconEntry
    modifiers   map[string]float64
    negations   map[string]bool
    mutex       sync.RWMutex
}

// LexiconEntry represents a word's sentiment information
type LexiconEntry struct {
    Word       string
    Sentiment  float64  // -1 to 1
    Confidence float64  // 0 to 1
    POS        []string // Applicable POS tags
    Domain     string   // Domain specificity
}

// LoadSentimentLexicon loads language-specific lexicon
func LoadSentimentLexicon(lang Language) *SentimentLexicon {
    lexicon := &SentimentLexicon{
        words:     make(map[string]LexiconEntry),
        modifiers: make(map[string]float64),
        negations: make(map[string]bool),
    }
    
    // Load base lexicon
    lexicon.loadBaseLexicon(lang)
    
    // Load modifiers
    lexicon.loadModifiers(lang)
    
    // Load negations
    lexicon.loadNegations(lang)
    
    return lexicon
}

// loadBaseLexicon loads sentiment words
func (sl *SentimentLexicon) loadBaseLexicon(lang Language) {
    // Example entries - would load from files
    baseWords := map[string]LexiconEntry{
        // Positive words
        "excellent": {Word: "excellent", Sentiment: 0.9, Confidence: 0.95},
        "good":      {Word: "good", Sentiment: 0.6, Confidence: 0.9},
        "love":      {Word: "love", Sentiment: 0.8, Confidence: 0.9},
        "amazing":   {Word: "amazing", Sentiment: 0.85, Confidence: 0.95},
        "wonderful": {Word: "wonderful", Sentiment: 0.85, Confidence: 0.95},
        "happy":     {Word: "happy", Sentiment: 0.7, Confidence: 0.9},
        "beautiful": {Word: "beautiful", Sentiment: 0.75, Confidence: 0.9},
        "fantastic": {Word: "fantastic", Sentiment: 0.85, Confidence: 0.95},
        
        // Negative words
        "terrible":     {Word: "terrible", Sentiment: -0.9, Confidence: 0.95},
        "bad":          {Word: "bad", Sentiment: -0.6, Confidence: 0.9},
        "hate":         {Word: "hate", Sentiment: -0.8, Confidence: 0.9},
        "awful":        {Word: "awful", Sentiment: -0.85, Confidence: 0.95},
        "horrible":     {Word: "horrible", Sentiment: -0.85, Confidence: 0.95},
        "sad":          {Word: "sad", Sentiment: -0.7, Confidence: 0.9},
        "ugly":         {Word: "ugly", Sentiment: -0.75, Confidence: 0.9},
        "disappointing":{Word: "disappointing", Sentiment: -0.7, Confidence: 0.9},
        
        // Context-dependent
        "cheap":     {Word: "cheap", Sentiment: -0.3, Confidence: 0.6}, // Can be positive in pricing
        "simple":    {Word: "simple", Sentiment: 0.1, Confidence: 0.5},  // Can be negative or positive
        "long":      {Word: "long", Sentiment: 0.0, Confidence: 0.3},   // Highly context-dependent
    }
    
    sl.words = baseWords
}

// loadModifiers loads intensifiers and diminishers
func (sl *SentimentLexicon) loadModifiers(lang Language) {
    sl.modifiers = map[string]float64{
        // Intensifiers (increase by factor)
        "very":      0.3,
        "extremely": 0.5,
        "absolutely":0.5,
        "totally":   0.4,
        "really":    0.3,
        "so":        0.3,
        "quite":     0.2,
        
        // Diminishers (decrease by factor)
        "slightly":  -0.3,
        "somewhat":  -0.3,
        "rather":    -0.2,
        "fairly":    -0.1,
        "marginally":-0.4,
        "barely":    -0.5,
        "hardly":    -0.5,
    }
}

// loadNegations loads negation words
func (sl *SentimentLexicon) loadNegations(lang Language) {
    sl.negations = map[string]bool{
        "not":    true,
        "no":     true,
        "never":  true,
        "neither":true,
        "nor":    true,
        "cannot": true,
        "can't":  true,
        "won't":  true,
        "don't":  true,
        "doesn't":true,
        "didn't": true,
        "isn't":  true,
        "aren't": true,
        "wasn't": true,
        "weren't":true,
        "without":true,
        "hardly": true,
        "scarcely":true,
        "barely": true,
    }
}

// GetSentiment returns sentiment score for a word
func (sl *SentimentLexicon) GetSentiment(word string) float64 {
    sl.mutex.RLock()
    defer sl.mutex.RUnlock()
    
    if entry, exists := sl.words[word]; exists {
        return entry.Sentiment
    }
    return 0.0
}

// GetConfidence returns confidence for a word's sentiment
func (sl *SentimentLexicon) GetConfidence(word string) float64 {
    sl.mutex.RLock()
    defer sl.mutex.RUnlock()
    
    if entry, exists := sl.words[word]; exists {
        return entry.Confidence
    }
    return 0.0
}

// IsNegation checks if word is a negation
func (sl *SentimentLexicon) IsNegation(word string) bool {
    sl.mutex.RLock()
    defer sl.mutex.RUnlock()
    
    return sl.negations[word]
}

// GetModifierStrength returns modifier strength
func (sl *SentimentLexicon) GetModifierStrength(word string) float64 {
    sl.mutex.RLock()
    defer sl.mutex.RUnlock()
    
    if strength, exists := sl.modifiers[word]; exists {
        return strength
    }
    return 0.0
}

// AddCustomWord allows adding domain-specific words
func (sl *SentimentLexicon) AddCustomWord(word string, sentiment, confidence float64) {
    sl.mutex.Lock()
    defer sl.mutex.Unlock()
    
    sl.words[word] = LexiconEntry{
        Word:       word,
        Sentiment:  sentiment,
        Confidence: confidence,
        Domain:     "custom",
    }
}
```

### 3. Feature Extraction (sentiment_features.go)

```go
package prose

import (
    "strings"
    "unicode"
)

// sentimentFeatureExtractor extracts features for ML classification
type sentimentFeatureExtractor struct {
    ngramSize   int
    usePOS      bool
    useDependency bool
}

// extractFeatures extracts features from tokens
func (sfe *sentimentFeatureExtractor) extractFeatures(tokens []*Token) map[string]float64 {
    features := make(map[string]float64)
    
    // 1. N-gram features
    sfe.extractNGrams(tokens, features)
    
    // 2. POS pattern features
    if sfe.usePOS {
        sfe.extractPOSPatterns(tokens, features)
    }
    
    // 3. Lexical features
    sfe.extractLexicalFeatures(tokens, features)
    
    // 4. Syntactic features
    sfe.extractSyntacticFeatures(tokens, features)
    
    // 5. Semantic features
    sfe.extractSemanticFeatures(tokens, features)
    
    return features
}

// extractNGrams extracts word n-grams
func (sfe *sentimentFeatureExtractor) extractNGrams(tokens []*Token, features map[string]float64) {
    // Unigrams
    for _, token := range tokens {
        key := "unigram:" + strings.ToLower(token.Text)
        features[key] = 1.0
    }
    
    // Bigrams
    for i := 0; i < len(tokens)-1; i++ {
        bigram := strings.ToLower(tokens[i].Text) + "_" + strings.ToLower(tokens[i+1].Text)
        key := "bigram:" + bigram
        features[key] = 1.0
    }
    
    // Trigrams
    if sfe.ngramSize >= 3 {
        for i := 0; i < len(tokens)-2; i++ {
            trigram := strings.ToLower(tokens[i].Text) + "_" + 
                      strings.ToLower(tokens[i+1].Text) + "_" + 
                      strings.ToLower(tokens[i+2].Text)
            key := "trigram:" + trigram
            features[key] = 1.0
        }
    }
}

// extractPOSPatterns extracts POS-based patterns
func (sfe *sentimentFeatureExtractor) extractPOSPatterns(tokens []*Token, features map[string]float64) {
    // POS sequences
    for i := 0; i < len(tokens)-1; i++ {
        pattern := tokens[i].Tag + "_" + tokens[i+1].Tag
        key := "pos_bigram:" + pattern
        features[key] = 1.0
    }
    
    // Adjective-Noun patterns
    for i := 0; i < len(tokens)-1; i++ {
        if strings.HasPrefix(tokens[i].Tag, "JJ") && strings.HasPrefix(tokens[i+1].Tag, "NN") {
            key := "adj_noun:" + strings.ToLower(tokens[i].Text) + "_" + strings.ToLower(tokens[i+1].Text)
            features[key] = 1.0
        }
    }
    
    // Verb-Adverb patterns
    for i := 0; i < len(tokens)-1; i++ {
        if strings.HasPrefix(tokens[i].Tag, "VB") && strings.HasPrefix(tokens[i+1].Tag, "RB") {
            key := "verb_adv:" + strings.ToLower(tokens[i].Text) + "_" + strings.ToLower(tokens[i+1].Text)
            features[key] = 1.0
        }
    }
}

// extractLexicalFeatures extracts word-level features
func (sfe *sentimentFeatureExtractor) extractLexicalFeatures(tokens []*Token, features map[string]float64) {
    var (
        exclamationCount int
        questionCount    int
        capsCount        int
        elongatedCount   int
    )
    
    for _, token := range tokens {
        // Punctuation features
        if token.Text == "!" {
            exclamationCount++
        }
        if token.Text == "?" {
            questionCount++
        }
        
        // Capitalization (excluding first word)
        if isAllCaps(token.Text) && len(token.Text) > 1 {
            capsCount++
        }
        
        // Elongated words (e.g., "sooooo")
        if isElongated(token.Text) {
            elongatedCount++
        }
    }
    
    // Normalize by token count
    tokenCount := float64(len(tokens))
    if tokenCount > 0 {
        features["exclamation_ratio"] = float64(exclamationCount) / tokenCount
        features["question_ratio"] = float64(questionCount) / tokenCount
        features["caps_ratio"] = float64(capsCount) / tokenCount
        features["elongated_ratio"] = float64(elongatedCount) / tokenCount
    }
    
    // Text length features
    features["token_count"] = tokenCount
    features["avg_word_length"] = calculateAvgWordLength(tokens)
}

// extractSyntacticFeatures extracts syntax-based features
func (sfe *sentimentFeatureExtractor) extractSyntacticFeatures(tokens []*Token, features map[string]float64) {
    var (
        negationCount    int
        comparativeCount int
        superlativeCount int
    )
    
    for _, token := range tokens {
        // Negation detection
        if isNegation(token.Text) {
            negationCount++
        }
        
        // Comparative and superlative adjectives
        if token.Tag == "JJR" { // Comparative
            comparativeCount++
        }
        if token.Tag == "JJS" { // Superlative
            superlativeCount++
        }
    }
    
    features["negation_count"] = float64(negationCount)
    features["comparative_count"] = float64(comparativeCount)
    features["superlative_count"] = float64(superlativeCount)
    
    // Dependency-based features would go here if dependency parsing is available
}

// extractSemanticFeatures extracts meaning-based features
func (sfe *sentimentFeatureExtractor) extractSemanticFeatures(tokens []*Token, features map[string]float64) {
    // Sentiment word counts
    var (
        posWordCount   int
        negWordCount   int
        neutralCount   int
    )
    
    // This would use the lexicon to count sentiment words
    // Implementation would depend on lexicon integration
    
    features["pos_word_ratio"] = float64(posWordCount) / float64(len(tokens))
    features["neg_word_ratio"] = float64(negWordCount) / float64(len(tokens))
    features["neutral_ratio"] = float64(neutralCount) / float64(len(tokens))
    
    // Emotion categories (would require emotion lexicon)
    features["has_joy_words"] = 0.0
    features["has_anger_words"] = 0.0
    features["has_fear_words"] = 0.0
    features["has_sadness_words"] = 0.0
    features["has_surprise_words"] = 0.0
}

// Helper functions

func isAllCaps(text string) bool {
    if len(text) == 0 {
        return false
    }
    for _, r := range text {
        if unicode.IsLetter(r) && !unicode.IsUpper(r) {
            return false
        }
    }
    return true
}

func isElongated(text string) bool {
    // Check for repeated characters (3+ in a row)
    if len(text) < 3 {
        return false
    }
    
    count := 1
    for i := 1; i < len(text); i++ {
        if text[i] == text[i-1] {
            count++
            if count >= 3 {
                return true
            }
        } else {
            count = 1
        }
    }
    return false
}

func calculateAvgWordLength(tokens []*Token) float64 {
    if len(tokens) == 0 {
        return 0
    }
    
    totalLength := 0
    wordCount := 0
    
    for _, token := range tokens {
        if isWord(token.Text) {
            totalLength += len(token.Text)
            wordCount++
        }
    }
    
    if wordCount == 0 {
        return 0
    }
    
    return float64(totalLength) / float64(wordCount)
}

func isWord(text string) bool {
    for _, r := range text {
        if unicode.IsLetter(r) {
            return true
        }
    }
    return false
}
```

### 4. ML Classifier Integration

```go
// sentimentClassifier wraps the ML model for sentiment
type sentimentClassifier struct {
    model    *binaryMaxentClassifier
    features *sentimentFeatureExtractor
}

// predict returns sentiment predictions with confidence
func (sc *sentimentClassifier) predict(features map[string]float64) SentimentScore {
    // Convert features to model format
    vector := sc.featuresToVector(features)
    
    // Get predictions for each class
    probs := sc.model.predictProba(vector)
    
    // Find dominant class
    maxProb := 0.0
    var dominant SentimentClass
    for class, prob := range probs {
        if prob > maxProb {
            maxProb = prob
            dominant = SentimentClass(class)
        }
    }
    
    // Calculate polarity and intensity from class probabilities
    polarity := sc.calculatePolarity(probs)
    intensity := sc.calculateIntensity(probs)
    
    return SentimentScore{
        Polarity:   polarity,
        Intensity:  intensity,
        Confidence: maxProb,
        Dominant:   dominant,
        Scores:     probs,
    }
}

// calculatePolarity derives polarity from class probabilities
func (sc *sentimentClassifier) calculatePolarity(probs map[SentimentClass]float64) float64 {
    positive := probs[StrongPositive]*1.0 + probs[Positive]*0.5
    negative := probs[StrongNegative]*1.0 + probs[Negative]*0.5
    
    if positive+negative == 0 {
        return 0
    }
    
    return (positive - negative) / (positive + negative)
}

// calculateIntensity derives intensity from class probabilities
func (sc *sentimentClassifier) calculateIntensity(probs map[SentimentClass]float64) float64 {
    strong := probs[StrongPositive] + probs[StrongNegative]
    mild := probs[Positive] + probs[Negative]
    neutral := probs[Neutral]
    
    // Weight strong sentiment higher
    return (strong*1.0 + mild*0.5) / (strong + mild + neutral + 0.00001)
}
```

## API Design

### Current Implementation ✅

**Core Sentiment Analysis API** (Available Now):

```go
// sentiment.go - IMPLEMENTED

// Create standard sentiment analyzer
func NewSentimentAnalyzer(lang Language, config SentimentConfig) *SentimentAnalyzer

// Create analyzer with external lexicon support ✅ NEW
func NewSentimentAnalyzerWithExternal(lang Language, config SentimentConfig, externalLexiconPath string) (*SentimentAnalyzer, error)

// Analyze document sentiment
func (sa *SentimentAnalyzer) AnalyzeDocument(doc *Document) SentimentScore

// Analyze sentence sentiment
func (sa *SentimentAnalyzer) AnalyzeSentence(tokens []*Token) SentimentScore

// Configuration
func DefaultSentimentConfig() SentimentConfig
```

**External Lexicon API** (Available Now):

```go
// sentiment_lexicon.go - IMPLEMENTED

// Load lexicon with external file support ✅ NEW
func LoadSentimentLexiconWithExternal(lang Language, externalPath string) (*SentimentLexicon, error)

// Load external lexicon at runtime ✅ NEW
func (sl *SentimentLexicon) LoadExternalLexicon(filepath string, languages []Language) error

// Standard lexicon operations
func (sl *SentimentLexicon) GetSentiment(word string) float64
func (sl *SentimentLexicon) IsModifier(word string) (float64, bool)
func (sl *SentimentLexicon) IsNegation(word string) bool
```

### Future Document-Level API (Phase 3)

```go
// Add to document.go - FUTURE IMPLEMENTATION

// WithSentiment enables sentiment analysis during document processing
func WithSentiment(enable bool) DocOpt {
    return func(doc *Document, opts *DocOpts) {
        opts.Sentiment = enable
    }
}

// Add to Document struct
type Document struct {
    // ... existing fields ...
    sentiment         *SentimentScore          // Document-level sentiment
    sentenceSentiment []SentimentScore        // Sentence-level sentiments
    aspectSentiment   []AspectSentiment       // Aspect-based sentiments
}

// Sentiment returns document-level sentiment
func (doc *Document) Sentiment() *SentimentScore {
    return doc.sentiment
}

// SentenceSentiments returns sentence-level sentiments
func (doc *Document) SentenceSentiments() []SentimentScore {
    return doc.sentenceSentiment
}

// AspectSentiments returns aspect-based sentiments
func (doc *Document) AspectSentiments() []AspectSentiment {
    return doc.aspectSentiment
}
```

### Usage Examples

```go
// Basic usage
doc, _ := prose.NewDocument("This movie is absolutely fantastic! I loved every minute of it.",
    prose.WithSentiment(true))

sentiment := doc.Sentiment()
fmt.Printf("Polarity: %.2f\n", sentiment.Polarity)       // 0.85
fmt.Printf("Intensity: %.2f\n", sentiment.Intensity)     // 0.90
fmt.Printf("Confidence: %.2f\n", sentiment.Confidence)   // 0.92
fmt.Printf("Class: %s\n", sentiment.Dominant)            // "strong_positive"

// Sentence-level analysis
for i, sent := range doc.Sentences() {
    sentScore := doc.SentenceSentiments()[i]
    fmt.Printf("Sentence %d: %s (%.2f)\n", i, sent.Text, sentScore.Polarity)
}

// Feature inspection
for _, word := range sentiment.Features.PositiveWords {
    fmt.Printf("Positive: %s (%.2f)\n", word.Word, word.AdjustedScore)
}

// Custom configuration
config := prose.DefaultSentimentConfig()
config.AspectAnalysis = true
config.MinConfidence = 0.7

analyzer := prose.NewSentimentAnalyzer(prose.English, config)
doc, _ := prose.NewDocument("The food was excellent but the service was terrible.",
    prose.UsingModel(&prose.Model{
        sentimentAnalyzer: analyzer,
    }))

// Aspect-based results
for _, aspect := range doc.AspectSentiments() {
    fmt.Printf("Aspect: %s - %s (%.2f)\n", 
        aspect.Aspect, aspect.Sentiment.Dominant, aspect.Sentiment.Polarity)
    // Output:
    // Aspect: food - positive (0.80)
    // Aspect: service - negative (-0.85)
}

// Adding custom domain words
analyzer.Lexicon().AddCustomWord("bullish", 0.7, 0.9)  // Financial domain
analyzer.Lexicon().AddCustomWord("bearish", -0.7, 0.9)

// Batch processing with progress
docs := []string{
    "Great product, highly recommend!",
    "Disappointed with the quality.",
    "Average experience, nothing special.",
}

results := make([]prose.SentimentScore, len(docs))
for i, text := range docs {
    doc, _ := prose.NewDocument(text, 
        prose.WithSentiment(true),
        prose.WithProgressCallback(func(p float64) {
            fmt.Printf("Processing doc %d: %.0f%%\n", i, p*100)
        }))
    results[i] = *doc.Sentiment()
}
```

## Testing Strategy

### Unit Tests (sentiment_test.go)

```go
package prose

import (
    "testing"
    "math"
)

func TestSentimentPolarity(t *testing.T) {
    tests := []struct {
        text     string
        expected float64
        delta    float64
    }{
        {"I love this product!", 0.8, 0.2},
        {"This is terrible.", -0.8, 0.2},
        {"It's okay.", 0.0, 0.2},
        {"Not bad at all.", 0.4, 0.2},  // Negation handling
        {"I don't like it.", -0.6, 0.2},
    }
    
    for _, tt := range tests {
        doc, err := NewDocument(tt.text, WithSentiment(true))
        if err != nil {
            t.Errorf("Failed to create document: %v", err)
            continue
        }
        
        sentiment := doc.Sentiment()
        if math.Abs(sentiment.Polarity-tt.expected) > tt.delta {
            t.Errorf("Text: %s\nExpected: %.2f\nGot: %.2f", 
                tt.text, tt.expected, sentiment.Polarity)
        }
    }
}

func TestSentimentIntensity(t *testing.T) {
    tests := []struct {
        text     string
        minIntensity float64
    }{
        {"This is absolutely amazing!", 0.8},
        {"It's very very bad.", 0.7},
        {"Slightly disappointing.", 0.3},
        {"TERRIBLE!!!", 0.9},
    }
    
    for _, tt := range tests {
        doc, _ := NewDocument(tt.text, WithSentiment(true))
        sentiment := doc.Sentiment()
        
        if sentiment.Intensity < tt.minIntensity {
            t.Errorf("Text: %s\nExpected intensity >= %.2f\nGot: %.2f",
                tt.text, tt.minIntensity, sentiment.Intensity)
        }
    }
}

func TestNegationHandling(t *testing.T) {
    pairs := []struct {
        positive string
        negated  string
    }{
        {"This is good.", "This is not good."},
        {"I like it.", "I don't like it."},
        {"Happy with the service.", "Not happy with the service."},
    }
    
    for _, pair := range pairs {
        posDoc, _ := NewDocument(pair.positive, WithSentiment(true))
        negDoc, _ := NewDocument(pair.negated, WithSentiment(true))
        
        posSent := posDoc.Sentiment()
        negSent := negDoc.Sentiment()
        
        // Negation should flip the polarity
        if posSent.Polarity * negSent.Polarity >= 0 {
            t.Errorf("Negation not handled properly:\n%s: %.2f\n%s: %.2f",
                pair.positive, posSent.Polarity,
                pair.negated, negSent.Polarity)
        }
    }
}

func TestModifierEffects(t *testing.T) {
    base := "This is good."
    intensified := "This is very good."
    diminished := "This is slightly good."
    
    baseDoc, _ := NewDocument(base, WithSentiment(true))
    intDoc, _ := NewDocument(intensified, WithSentiment(true))
    dimDoc, _ := NewDocument(diminished, WithSentiment(true))
    
    baseSent := baseDoc.Sentiment()
    intSent := intDoc.Sentiment()
    dimSent := dimDoc.Sentiment()
    
    // Intensifier should increase magnitude
    if math.Abs(intSent.Polarity) <= math.Abs(baseSent.Polarity) {
        t.Errorf("Intensifier failed: base=%.2f, intensified=%.2f",
            baseSent.Polarity, intSent.Polarity)
    }
    
    // Diminisher should decrease magnitude
    if math.Abs(dimSent.Polarity) >= math.Abs(baseSent.Polarity) {
        t.Errorf("Diminisher failed: base=%.2f, diminished=%.2f",
            baseSent.Polarity, dimSent.Polarity)
    }
}

func TestAspectSentiment(t *testing.T) {
    text := "The food was delicious but the service was slow and the atmosphere was nice."
    
    config := DefaultSentimentConfig()
    config.AspectAnalysis = true
    
    analyzer := NewSentimentAnalyzer(English, config)
    doc, _ := NewDocument(text, 
        UsingModel(&Model{sentimentAnalyzer: analyzer}))
    
    aspects := doc.AspectSentiments()
    
    expectedAspects := map[string]float64{
        "food":       0.7,  // positive
        "service":   -0.5,  // negative
        "atmosphere": 0.5,  // positive
    }
    
    for _, aspect := range aspects {
        if expected, exists := expectedAspects[aspect.Aspect]; exists {
            if math.Abs(aspect.Sentiment.Polarity-expected) > 0.3 {
                t.Errorf("Aspect %s: expected %.2f, got %.2f",
                    aspect.Aspect, expected, aspect.Sentiment.Polarity)
            }
        }
    }
}

func TestMultilingualSentiment(t *testing.T) {
    tests := []struct {
        lang     Language
        text     string
        expected float64
    }{
        {English, "This is wonderful!", 0.8},
        {Spanish, "¡Esto es maravilloso!", 0.8},
        {French, "C'est merveilleux!", 0.8},
        {German, "Das ist wunderbar!", 0.8},
    }
    
    for _, tt := range tests {
        doc, _ := NewDocument(tt.text, 
            WithLanguage(tt.lang),
            WithSentiment(true))
        
        sentiment := doc.Sentiment()
        if math.Abs(sentiment.Polarity-tt.expected) > 0.3 {
            t.Errorf("Language %s: expected %.2f, got %.2f",
                tt.lang, tt.expected, sentiment.Polarity)
        }
    }
}

func BenchmarkSentimentAnalysis(b *testing.B) {
    texts := []string{
        "This product exceeded my expectations in every way.",
        "The customer service was absolutely terrible.",
        "It's an okay product, nothing special.",
        "I'm not sure if I like it or not.",
    }
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        doc, _ := NewDocument(texts[i%len(texts)], WithSentiment(true))
        _ = doc.Sentiment()
    }
}

func BenchmarkSentimentWithAspects(b *testing.B) {
    text := "The food was excellent, the service was poor, but the ambiance made up for it."
    
    config := DefaultSentimentConfig()
    config.AspectAnalysis = true
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        analyzer := NewSentimentAnalyzer(English, config)
        doc, _ := NewDocument(text, 
            UsingModel(&Model{sentimentAnalyzer: analyzer}))
        _ = doc.AspectSentiments()
    }
}
```

## Performance Considerations

### Memory Optimization
1. **Reuse Lexicon Instances**: Share lexicon across analyzer instances
2. **Token Pooling**: Leverage existing token pool infrastructure
3. **Feature Caching**: Cache frequently used feature vectors
4. **Lazy Loading**: Load language-specific resources on demand

### Speed Optimization
1. **Parallel Processing**: Process sentences concurrently for documents
2. **Early Termination**: Skip processing if confidence threshold not met
3. **Optimized Lookups**: Use hash maps for O(1) lexicon lookups
4. **Batch Processing**: Process multiple documents in batches

### Accuracy Optimization
1. **Ensemble Methods**: Combine lexicon and ML predictions
2. **Context Windows**: Adjust negation/modifier windows dynamically
3. **Domain Adaptation**: Allow custom lexicons for specific domains
4. **Confidence Calibration**: Calibrate confidence scores using validation data

## Integration Checklist

### Phase 1: Core Implementation ✅ COMPLETED
- [x] Create `sentiment.go` with base analyzer structure
- [x] Implement `sentiment_lexicon.go` with basic lexicons
- [x] Add type definitions to `types.go`
- [x] Create `sentiment_features.go` for feature extraction
- [x] Write unit tests in `sentiment_test.go`

**Additional Phase 1 Enhancements:**
- [x] **Hybrid Lexicon System**: External JSON file support for custom domain lexicons
- [x] **Advanced Linguistic Processing**: Negation handling, intensifiers, diminishers
- [x] **Multilingual Implementation**: Complete support for 5 languages with language-specific features
- [x] **Comprehensive Documentation**: EXTERNAL-LEXICON-GUIDE.md with examples and best practices

### Phase 2: Model Integration
- [ ] Integrate with existing `binaryMaxentClassifier`
- [ ] Add sentiment model to `Model` struct
- [ ] Implement model training in `training.go`
- [ ] Create model serialization/deserialization

### Phase 3: Document Integration
- [ ] Add `WithSentiment` option to document creation
- [ ] Integrate sentiment analysis into document processing pipeline
- [ ] Add sentiment fields to `Document` struct
- [ ] Update document tests

### Phase 4: Multilingual Support
- [ ] Create language-specific lexicons
- [ ] Add to `multilingual.go`
- [ ] Test with each supported language
- [ ] Document language-specific behavior

### Phase 5: Advanced Features
- [ ] Implement aspect-based sentiment analysis
- [ ] Add sarcasm/irony detection
- [ ] Implement emotion detection
- [ ] Add domain adaptation capabilities

### Phase 6: Documentation & Examples
- [ ] Update main README with sentiment examples
- [ ] Create sentiment-specific documentation
- [ ] Add example applications
- [ ] Write performance benchmarks

### Phase 7: Testing & Validation
- [ ] Unit tests with >80% coverage
- [ ] Integration tests with real documents
- [ ] Performance benchmarks
- [ ] Validation against standard datasets

### Phase 8: Final Polish
- [ ] Code review and refactoring
- [ ] Performance optimization
- [ ] Security review (no sensitive data in lexicons)
- [ ] Version update and changelog

## Conclusion

This implementation guide provides a comprehensive approach to adding sentiment analysis to the Prose NLP library. The design:

1. **Follows existing patterns** established in the codebase
2. **Reuses infrastructure** like confidence scoring and model management
3. **Maintains backward compatibility** while adding new features
4. **Provides flexibility** through configuration options
5. **Ensures production readiness** with proper testing and optimization

The implementation combines lexicon-based and machine learning approaches to provide accurate, confident sentiment analysis across multiple languages and domains. The modular design allows for easy extension and customization while maintaining the library's focus on performance and simplicity.