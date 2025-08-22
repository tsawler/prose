package prose

import (
	"math"
	"strings"
)

// SentimentAnalyzer performs sentiment analysis
type SentimentAnalyzer struct {
	lexicon    *SentimentLexicon
	classifier *sentimentClassifier
	config     SentimentConfig
}

// SentimentConfig configures sentiment analysis
type SentimentConfig struct {
	UseLexicon     bool
	UseML          bool
	UseContext     bool
	MinConfidence  float64
	NegationWindow int  // Words to check for negation
	AspectAnalysis bool
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

// NewSentimentAnalyzerWithExternal creates a sentiment analyzer with external lexicon support
func NewSentimentAnalyzerWithExternal(lang Language, config SentimentConfig, externalLexiconPath string) (*SentimentAnalyzer, error) {
	lexicon, err := LoadSentimentLexiconWithExternal(lang, externalLexiconPath)
	if err != nil {
		return nil, err
	}

	classifier, err := loadSentimentModelWithExternal(lang, externalLexiconPath)
	if err != nil {
		return nil, err
	}

	return &SentimentAnalyzer{
		lexicon:    lexicon,
		classifier: classifier,
		config:     config,
	}, nil
}

// AnalyzeDocument performs document-level sentiment analysis
func (sa *SentimentAnalyzer) AnalyzeDocument(doc *Document) SentimentScore {
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
	if sa.config.UseML && sa.classifier != nil {
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
	var polarity float64
	if posScore == 0 && negScore == 0 {
		polarity = 0
	} else if negScore == 0 {
		// Only positive sentiment - scale based on strength
		polarity = math.Min(1.0, posScore*1.5) // Allow up to 1.0 for very strong positive
	} else if posScore == 0 {
		// Only negative sentiment - scale based on strength
		polarity = math.Max(-1.0, -negScore*1.5) // Allow up to -1.0 for very strong negative
	} else {
		// Both positive and negative - normalize
		totalScore := posScore + negScore
		polarity = (posScore - negScore) / totalScore
	}

	// Calculate intensity (0 to 1)
	// Intensity should reflect the strength of sentiment regardless of polarity
	maxScore := math.Max(posScore, negScore)
	intensity := math.Min(1.0, maxScore*1.5) // Scale up intensity

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
		// Check for standard negations
		if sa.lexicon.IsNegation(tokens[i].Text) {
			// Check if there's a clause boundary between negation and target
			for j := i + 1; j < position; j++ {
				if isClauseBoundary(tokens[j]) {
					return false
				}
			}
			return true
		}

		// Check for contraction negations like "n't"
		if tokens[i].Text == "n't" || tokens[i].Text == "not" {
			// Check if there's a clause boundary between negation and target
			for j := i + 1; j < position; j++ {
				if isClauseBoundary(tokens[j]) {
					return false
				}
			}
			return true
		}

		// Check for combined tokens like "don't", "can't", etc.
		lowerText := strings.ToLower(tokens[i].Text)
		if strings.Contains(lowerText, "n't") || sa.lexicon.IsNegation(lowerText) {
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
	if position == 0 || baseSentiment == 0 {
		return baseSentiment
	}

	// Check previous 2 tokens for modifiers
	start := maxInt(0, position-2)
	
	for i := start; i < position; i++ {
		modifier := sa.lexicon.GetModifierStrength(tokens[i].Text)
		if modifier != 0 {
			if modifier > 0 {
				// Intensifier - increase magnitude
				return baseSentiment * (1 + modifier)
			} else {
				// Diminisher - decrease magnitude
				return baseSentiment * (1 + modifier) // modifier is negative
			}
		}
	}

	return baseSentiment
}

// extractFeatures extracts features for ML classification
func (sa *SentimentAnalyzer) extractFeatures(tokens []*Token) map[string]float64 {
	if sa.classifier == nil || sa.classifier.features == nil {
		return make(map[string]float64)
	}
	return sa.classifier.features.extractFeatures(tokens)
}

// applyContextualRules applies context-based adjustments
func (sa *SentimentAnalyzer) applyContextualRules(score SentimentScore, tokens []*Token) SentimentScore {
	// Check for contradictions
	hasPositive := len(score.Features.PositiveWords) > 0
	hasNegative := len(score.Features.NegativeWords) > 0

	if hasPositive && hasNegative {
		// Mixed sentiment detection
		posStrength := 0.0
		negStrength := 0.0

		for _, word := range score.Features.PositiveWords {
			posStrength += math.Abs(word.AdjustedScore)
		}
		for _, word := range score.Features.NegativeWords {
			negStrength += math.Abs(word.AdjustedScore)
		}

		// If strengths are similar, mark as mixed
		ratio := math.Min(posStrength, negStrength) / math.Max(posStrength, negStrength)
		if ratio > 0.7 {
			score.Dominant = Mixed
			score.Confidence *= 0.8 // Reduce confidence for mixed sentiment
		}
	}

	// Check for question marks (often reduce sentiment certainty)
	for _, token := range tokens {
		if token.Text == "?" {
			score.Confidence *= 0.9
			score.Intensity *= 0.9
			break
		}
	}

	return score
}

// aggregateSentiments combines sentence-level sentiments
func (sa *SentimentAnalyzer) aggregateSentiments(sentenceScores []SentimentScore) SentimentScore {
	if len(sentenceScores) == 0 {
		return SentimentScore{
			Polarity:   0,
			Intensity:  0,
			Confidence: 0,
			Dominant:   Neutral,
		}
	}

	var (
		totalPolarity   float64
		totalIntensity  float64
		totalConfidence float64
		weights         float64
	)

	// Weighted average based on confidence
	for _, score := range sentenceScores {
		weight := score.Confidence
		totalPolarity += score.Polarity * weight
		totalIntensity += score.Intensity * weight
		totalConfidence += score.Confidence
		weights += weight
	}

	if weights == 0 {
		weights = 1
	}

	avgPolarity := totalPolarity / weights
	avgIntensity := totalIntensity / weights
	avgConfidence := totalConfidence / float64(len(sentenceScores))

	// Combine features from all sentences
	var features SentimentFeatures
	for _, score := range sentenceScores {
		features.PositiveWords = append(features.PositiveWords, score.Features.PositiveWords...)
		features.NegativeWords = append(features.NegativeWords, score.Features.NegativeWords...)
		features.Modifiers = append(features.Modifiers, score.Features.Modifiers...)
		features.Negations = append(features.Negations, score.Features.Negations...)
		features.Intensifiers = append(features.Intensifiers, score.Features.Intensifiers...)
	}

	dominant := classifyPolarity(avgPolarity, avgIntensity)

	return SentimentScore{
		Polarity:   avgPolarity,
		Intensity:  avgIntensity,
		Confidence: avgConfidence,
		Dominant:   dominant,
		Features:   features,
		Scores: map[SentimentClass]float64{
			StrongPositive: calculateClassProb(avgPolarity, avgIntensity, StrongPositive),
			Positive:       calculateClassProb(avgPolarity, avgIntensity, Positive),
			Neutral:        calculateClassProb(avgPolarity, avgIntensity, Neutral),
			Negative:       calculateClassProb(avgPolarity, avgIntensity, Negative),
			StrongNegative: calculateClassProb(avgPolarity, avgIntensity, StrongNegative),
		},
	}
}

// Helper functions

// extractSentenceTokens extracts tokens belonging to a sentence
func extractSentenceTokens(sent Sentence, tokens []Token) []*Token {
	var sentTokens []*Token
	for i := range tokens {
		if tokens[i].Start >= sent.Start && tokens[i].End <= sent.End {
			sentTokens = append(sentTokens, &tokens[i])
		}
	}
	return sentTokens
}

// isContentWord checks if a token is a content word
func isContentWord(token *Token) bool {
	// Skip punctuation and very short words
	if len(token.Text) <= 1 {
		return false
	}
	// Check if it's a meaningful POS tag
	if token.Tag != "" {
		// Content words: nouns, verbs, adjectives, adverbs
		return strings.HasPrefix(token.Tag, "NN") ||
			strings.HasPrefix(token.Tag, "VB") ||
			strings.HasPrefix(token.Tag, "JJ") ||
			strings.HasPrefix(token.Tag, "RB")
	}
	// If no POS tag, accept words with letters
	for _, r := range token.Text {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			return true
		}
	}
	return false
}

// isClauseBoundary checks if a token represents a clause boundary
func isClauseBoundary(token *Token) bool {
	boundaries := map[string]bool{
		",":   true,
		";":   true,
		":":   true,
		".":   true,
		"!":   true,
		"?":   true,
		"but": true,
		"however": true,
		"although": true,
	}
	return boundaries[strings.ToLower(token.Text)]
}

// classifyPolarity determines the sentiment class from polarity and intensity
func classifyPolarity(polarity, intensity float64) SentimentClass {
	if math.Abs(polarity) < 0.1 {
		return Neutral
	}
	
	if polarity > 0 {
		if intensity > 0.6 && polarity > 0.5 {
			return StrongPositive
		}
		return Positive
	}
	
	if intensity > 0.6 && polarity < -0.5 {
		return StrongNegative
	}
	return Negative
}

// calculateClassProb calculates probability for a sentiment class
func calculateClassProb(polarity, intensity float64, class SentimentClass) float64 {
	// Simple gaussian-like distribution around class centers
	var center, spread float64
	
	switch class {
	case StrongPositive:
		center = 0.8
		spread = 0.2
	case Positive:
		center = 0.4
		spread = 0.3
	case Neutral:
		center = 0.0
		spread = 0.2
	case Negative:
		center = -0.4
		spread = 0.3
	case StrongNegative:
		center = -0.8
		spread = 0.2
	default:
		return 0
	}
	
	// Calculate distance from center
	distance := math.Abs(polarity - center)
	
	// Gaussian-like probability
	prob := math.Exp(-distance * distance / (2 * spread * spread))
	
	// Adjust for intensity
	if class == StrongPositive || class == StrongNegative {
		prob *= intensity
	} else if class == Neutral {
		prob *= (1 - intensity)
	}
	
	return math.Min(1.0, math.Max(0.0, prob))
}

// combineScores combines two sentiment scores with weighting
func combineScores(score1, score2 SentimentScore, weight2 float64) SentimentScore {
	weight1 := 1.0 - weight2
	
	// If first score is empty, return second
	if score1.Confidence == 0 {
		return score2
	}
	
	combined := SentimentScore{
		Polarity:     score1.Polarity*weight1 + score2.Polarity*weight2,
		Intensity:    score1.Intensity*weight1 + score2.Intensity*weight2,
		Confidence:   score1.Confidence*weight1 + score2.Confidence*weight2,
		Subjectivity: score1.Subjectivity*weight1 + score2.Subjectivity*weight2,
	}
	
	// Combine features
	combined.Features.PositiveWords = append(score1.Features.PositiveWords, score2.Features.PositiveWords...)
	combined.Features.NegativeWords = append(score1.Features.NegativeWords, score2.Features.NegativeWords...)
	combined.Features.Modifiers = append(score1.Features.Modifiers, score2.Features.Modifiers...)
	combined.Features.Negations = append(score1.Features.Negations, score2.Features.Negations...)
	combined.Features.Intensifiers = append(score1.Features.Intensifiers, score2.Features.Intensifiers...)
	
	// Recalculate dominant class
	combined.Dominant = classifyPolarity(combined.Polarity, combined.Intensity)
	
	// Combine class scores
	combined.Scores = make(map[SentimentClass]float64)
	for class := range score1.Scores {
		combined.Scores[class] = score1.Scores[class]*weight1 + score2.Scores[class]*weight2
	}
	
	return combined
}

// maxInt returns the maximum of two integers
func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}

// sentimentClassifier wraps the ML model for sentiment
type sentimentClassifier struct {
	model    *binaryMaxentClassifier
	features *sentimentFeatureExtractor
}

// predict returns sentiment predictions with confidence
func (sc *sentimentClassifier) predict(features map[string]float64) SentimentScore {
	// For now, return empty score since ML model isn't loaded
	// This will be implemented in Phase 2
	return SentimentScore{
		Polarity:   0,
		Intensity:  0,
		Confidence: 0,
		Dominant:   Neutral,
		Scores: map[SentimentClass]float64{
			StrongPositive: 0,
			Positive:       0,
			Neutral:        1,
			Negative:       0,
			StrongNegative: 0,
		},
	}
}

// loadSentimentModel loads a sentiment classification model
func loadSentimentModel(lang Language) *sentimentClassifier {
	// For Phase 1, return basic classifier with feature extractor
	return &sentimentClassifier{
		model:    nil, // Will be implemented in Phase 2
		features: newSentimentFeatureExtractor(lang),
	}
}

// loadSentimentModelWithExternal loads a sentiment model with external lexicon support
func loadSentimentModelWithExternal(lang Language, externalPath string) (*sentimentClassifier, error) {
	features, err := newSentimentFeatureExtractorWithExternal(lang, externalPath)
	if err != nil {
		return nil, err
	}

	return &sentimentClassifier{
		model:    nil, // Will be implemented in Phase 2
		features: features,
	}, nil
}