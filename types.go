package prose

import (
	"context"
	"sync"
	"time"
)

// A Token represents an individual token of text such as a word or punctuation
// symbol.
type Token struct {
	Tag        string  // The token's part-of-speech tag.
	Text       string  // The token's actual content.
	Label      string  // The token's IOB label.
	Start      int     // Start position in original text
	End        int     // End position in original text
	Confidence float64 // Confidence score for predictions (0.0-1.0)
}

// An Entity represents an individual named-entity.
type Entity struct {
	Text       string  // The entity's actual content.
	Label      string  // The entity's label.
	Start      int     // Start position in original text
	End        int     // End position in original text
	Confidence float64 // Confidence score for entity prediction
}

// A Sentence represents a segmented portion of text.
type Sentence struct {
	Text  string // The sentence's text.
	Start int    // Start position in original text
	End   int    // End position in original text
}

// String returns the text content of the sentence
func (s Sentence) String() string {
	return s.Text
}

// ProcessingOptions contains configuration for document processing
type ProcessingOptions struct {
	Context         context.Context
	Timeout         time.Duration
	MaxTokens       int
	ProgressCallback func(progress float64)
	EnableCaching    bool
}

// TokenPool manages a pool of Token objects to reduce GC pressure
type TokenPool struct {
	pool sync.Pool
}

// NewTokenPool creates a new token pool
func NewTokenPool() *TokenPool {
	return &TokenPool{
		pool: sync.Pool{
			New: func() interface{} {
				return &Token{}
			},
		},
	}
}

// Get retrieves a token from the pool
func (tp *TokenPool) Get() *Token {
	return tp.pool.Get().(*Token)
}

// Put returns a token to the pool
func (tp *TokenPool) Put(token *Token) {
	// Reset token fields
	token.Tag = ""
	token.Text = ""
	token.Label = ""
	token.Start = 0
	token.End = 0
	token.Confidence = 0.0
	tp.pool.Put(token)
}

// Language represents supported languages
type Language string

const (
	English  Language = "en"
	Spanish  Language = "es"
	French   Language = "fr"
	German   Language = "de"
	Japanese Language = "ja"
)

// DocumentMetadata contains metadata about processed documents
type DocumentMetadata struct {
	Language     Language
	ProcessedAt  time.Time
	ProcessingTimeMs int64
	TokenCount   int
	SentenceCount int
	EntityCount  int
	Version      string
}

// ConfidenceLevel represents different confidence thresholds
type ConfidenceLevel float64

const (
	LowConfidence    ConfidenceLevel = 0.5
	MediumConfidence ConfidenceLevel = 0.7
	HighConfidence   ConfidenceLevel = 0.9
)

// EntityType represents different types of named entities
type EntityType string

const (
	PersonEntity       EntityType = "PERSON"       // People, including fictional
	OrganizationEntity EntityType = "ORG"          // Companies, agencies, institutions
	LocationEntity     EntityType = "GPE"          // Geopolitical entities (countries, cities, states)
	MoneyEntity        EntityType = "MONEY"        // Monetary values
	DateEntity         EntityType = "DATE"         // Dates
	TimeEntity         EntityType = "TIME"         // Times
	PercentEntity      EntityType = "PERCENT"      // Percentage values
	FacilityEntity     EntityType = "FAC"          // Buildings, airports, highways, bridges
	ProductEntity      EntityType = "PRODUCT"      // Objects, vehicles, foods (not services)
	EventEntity        EntityType = "EVENT"        // Named hurricanes, battles, wars, sports events
	WorkOfArtEntity    EntityType = "WORK_OF_ART"  // Titles of books, songs, etc.
	LanguageEntity     EntityType = "LANGUAGE"     // Any named language
	NorpEntity         EntityType = "NORP"         // Nationalities or religious or political groups
	LawEntity          EntityType = "LAW"          // Named documents made into laws
	OrdinalEntity      EntityType = "ORDINAL"      // "first", "second", etc.
	CardinalEntity     EntityType = "CARDINAL"     // Numerals that do not fall under another type
)

// SentimentScore represents the sentiment analysis results
type SentimentScore struct {
	// Primary sentiment metrics
	Polarity   float64 // -1.0 (negative) to 1.0 (positive)
	Intensity  float64 // 0.0 (neutral) to 1.0 (strong)
	Confidence float64 // 0.0 to 1.0 reliability score

	// Detailed classification
	Dominant SentimentClass             // Primary sentiment class
	Scores   map[SentimentClass]float64 // Individual class probabilities

	// Subjectivity analysis
	Subjectivity float64 // 0.0 (objective) to 1.0 (subjective)

	// Contributing factors
	Features SentimentFeatures // Key features affecting sentiment
}

// SentimentClass represents sentiment categories
type SentimentClass string

const (
	StrongPositive SentimentClass = "strong_positive"
	Positive       SentimentClass = "positive"
	Neutral        SentimentClass = "neutral"
	Negative       SentimentClass = "negative"
	StrongNegative SentimentClass = "strong_negative"
	Mixed          SentimentClass = "mixed" // For conflicting sentiments
)

// SentimentFeatures tracks contributing factors
type SentimentFeatures struct {
	PositiveWords []WordContribution
	NegativeWords []WordContribution
	Modifiers     []ModifierEffect
	Negations     []NegationScope
	Intensifiers  []IntensifierEffect
}

// WordContribution represents a word's sentiment contribution
type WordContribution struct {
	Word          string
	Position      int
	BaseScore     float64
	AdjustedScore float64
	Confidence    float64
}

// ModifierEffect represents sentiment modifiers
type ModifierEffect struct {
	Modifier string
	Type     ModifierType
	Position int
	Impact   float64
}

// ModifierType categorizes sentiment modifiers
type ModifierType string

const (
	Intensifier ModifierType = "intensifier" // "very", "extremely"
	Diminisher  ModifierType = "diminisher"  // "slightly", "somewhat"
	Negation    ModifierType = "negation"    // "not", "never"
)

// NegationScope represents the scope of a negation
type NegationScope struct {
	Position int
	Scope    int
}

// IntensifierEffect represents the effect of an intensifier
type IntensifierEffect struct {
	Word     string
	Position int
	Factor   float64
}

// AspectSentiment for aspect-based sentiment analysis
type AspectSentiment struct {
	Aspect    string
	Text      string
	Sentiment SentimentScore
	Start     int
	End       int
}

// SentimentContext provides analysis context
type SentimentContext struct {
	Domain  string // "general", "financial", "medical", etc.
	Sarcasm bool   // Sarcasm detection flag
	Irony   bool   // Irony detection flag
	Emoji   bool   // Consider emoji sentiment
}
