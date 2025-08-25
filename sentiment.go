package prose

import (
	"fmt"
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
		classifier: loadBasicSentimentModel(lang),
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

// NewSentimentAnalyzerFromModel creates a sentiment analyzer from a trained model
func NewSentimentAnalyzerFromModel(lang Language, config SentimentConfig, model *binaryMaxentClassifier) *SentimentAnalyzer {
	return &SentimentAnalyzer{
		lexicon:    LoadSentimentLexicon(lang),
		classifier: loadSentimentModelFromModel(lang, model),
		config:     config,
	}
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
	if sa.classifier == nil {
		return make(map[string]float64)
	}
	
	// Use simplified features that match training
	// Create a document from tokens for feature extraction
	text := ""
	for i, token := range tokens {
		if i > 0 {
			text += " "
		}
		text += token.Text
	}
	
	// Use fast tokenization instead of full NLP processing for training
	features := make(map[string]float64)
	
	// Simple word-level tokenization (much faster than full NLP)
	words := strings.Fields(text)
	docTokens := make([]Token, len(words))
	for i, word := range words {
		docTokens[i] = Token{Text: word}
	}
	
	// Simple unigram features
	wordCounts := make(map[string]int)
	for _, token := range docTokens {
		word := strings.ToLower(token.Text)
		if len(word) > 2 {
			wordCounts[word]++
		}
	}
	
	// Add word features
	for word, count := range wordCounts {
		if count > 0 {
			features["word:"+word] = float64(count)
		}
	}
	
	// Expanded intensity-aware sentiment indicators
	strongPosWords := []string{
		"amazing", "excellent", "outstanding", "brilliant", "perfect", "fantastic", "awesome", "superb", 
		"magnificent", "exceptional", "love", "adore", "incredible", "phenomenal", "extraordinary", 
		"spectacular", "wonderful", "marvelous", "fabulous", "sensational", "stunning", "breathtaking",
		"flawless", "impeccable", "divine", "sublime", "exquisite", "astounding", "remarkable", "thrilled",
		"ecstatic", "overjoyed", "delighted", "best", "greatest", "finest", "ultimate", "supreme"}
	
	weakPosWords := []string{
		"good", "nice", "okay", "fine", "decent", "adequate", "satisfactory", "reasonable", "solid",
		"fair", "acceptable", "pleasant", "comfortable", "suitable", "workable", "functional"}
	
	strongNegWords := []string{
		"terrible", "awful", "horrible", "disgusting", "atrocious", "appalling", "pathetic", "useless", 
		"worthless", "hate", "despise", "abysmal", "dreadful", "ghastly", "hideous", "revolting",
		"repulsive", "vile", "despicable", "detestable", "loathe", "abhor", "worst", "nightmare",
		"disaster", "catastrophe", "garbage", "trash", "junk", "crap", "shit", "hell", "damn"}
	
	weakNegWords := []string{
		"bad", "poor", "disappointing", "mediocre", "subpar", "inferior", "lacking", "flawed",
		"unsatisfactory", "inadequate", "unacceptable", "problematic", "deficient", "faulty"}
	
	strongPosCount := 0
	weakPosCount := 0
	strongNegCount := 0
	weakNegCount := 0
	
	for _, token := range docTokens {
		word := strings.ToLower(token.Text)
		for _, w := range strongPosWords {
			if word == w {
				strongPosCount++
			}
		}
		for _, w := range weakPosWords {
			if word == w {
				weakPosCount++
			}
		}
		for _, w := range strongNegWords {
			if word == w {
				strongNegCount++
			}
		}
		for _, w := range weakNegWords {
			if word == w {
				weakNegCount++
			}
		}
	}
	
	features["strong_pos_count"] = float64(strongPosCount)
	features["weak_pos_count"] = float64(weakPosCount)
	features["strong_neg_count"] = float64(strongNegCount)
	features["weak_neg_count"] = float64(weakNegCount)
	
	// Overall sentiment polarity
	features["pos_count"] = float64(strongPosCount + weakPosCount)
	features["neg_count"] = float64(strongNegCount + weakNegCount)
	
	// Sentiment intensity ratio
	totalSentiment := float64(strongPosCount + weakPosCount + strongNegCount + weakNegCount)
	if totalSentiment > 0 {
		features["strong_sentiment_ratio"] = float64(strongPosCount + strongNegCount) / totalSentiment
	}
	
	// Enhanced distinguishing features for strong vs weak sentiment
	if strongNegCount > 0 && weakNegCount == 0 && strongPosCount == 0 {
		features["pure_strong_negative"] = 1.0
	}
	if strongPosCount > 0 && weakPosCount == 0 && strongNegCount == 0 {
		features["pure_strong_positive"] = 1.0
	}
	if weakNegCount > 0 && strongNegCount == 0 && strongPosCount == 0 {
		features["pure_weak_negative"] = 1.0
	}
	if weakPosCount > 0 && strongPosCount == 0 && strongNegCount == 0 {
		features["pure_weak_positive"] = 1.0
	}
	
	// Strong sentiment dominance patterns
	if strongNegCount > weakNegCount && strongNegCount > 0 {
		features["strong_neg_dominant"] = 1.0
	}
	if strongPosCount > weakPosCount && strongPosCount > 0 {
		features["strong_pos_dominant"] = 1.0
	}
	if weakNegCount > strongNegCount && weakNegCount > 0 {
		features["weak_neg_dominant"] = 1.0
	}
	if weakPosCount > strongPosCount && weakPosCount > 0 {
		features["weak_pos_dominant"] = 1.0
	}
	
	// Multiple strong words indicate very strong sentiment
	if strongNegCount > 1 {
		features["multiple_strong_negative"] = float64(strongNegCount)
	}
	if strongPosCount > 1 {
		features["multiple_strong_positive"] = float64(strongPosCount)
	}
	
	// Intensity ratios
	if strongNegCount + weakNegCount > 0 {
		features["strong_neg_ratio"] = float64(strongNegCount) / float64(strongNegCount + weakNegCount)
	}
	if strongPosCount + weakPosCount > 0 {
		features["strong_pos_ratio"] = float64(strongPosCount) / float64(strongPosCount + weakPosCount)
	}
	
	if strongPosCount == 0 && weakPosCount == 0 && strongNegCount == 0 && weakNegCount == 0 {
		features["no_sentiment_words"] = 1.0
	}
	
	features["length"] = float64(len(docTokens))
	
	// Punctuation and intensity features
	exclamations := 0
	questions := 0
	allCaps := 0
	
	for _, token := range docTokens {
		text := token.Text
		if strings.Contains(text, "!") {
			exclamations++
		}
		if strings.Contains(text, "?") {
			questions++
		}
		if len(text) > 2 && text == strings.ToUpper(text) && strings.ToLower(text) != text {
			allCaps++
		}
	}
	
	features["exclamations"] = float64(exclamations)
	features["questions"] = float64(questions)
	features["all_caps_words"] = float64(allCaps)
	
	// Enhanced intensity indicators
	multiExclamations := 0
	intensifiers := 0
	superlatives := 0
	
	intensifierWords := []string{"very", "extremely", "incredibly", "absolutely", "totally", "completely", 
		"utterly", "quite", "really", "truly", "highly", "deeply", "super", "so", "too", "way"}
	superlativeWords := []string{"most", "least", "best", "worst", "greatest", "smallest", "biggest", 
		"highest", "lowest", "finest", "ultimate", "maximum", "minimum"}
		
	for _, token := range docTokens {
		text := strings.ToLower(token.Text)
		
		// Multiple exclamation marks
		if strings.Count(token.Text, "!") > 1 {
			multiExclamations++
		}
		
		// Intensifier words
		for _, intensifier := range intensifierWords {
			if text == intensifier {
				intensifiers++
				break
			}
		}
		
		// Superlative words
		for _, superlative := range superlativeWords {
			if text == superlative {
				superlatives++
				break
			}
		}
	}
	
	features["multi_exclamations"] = float64(multiExclamations)
	features["intensifiers"] = float64(intensifiers)
	features["superlatives"] = float64(superlatives)
	
	// Text pattern features for intensity
	fullText := ""
	for i, token := range docTokens {
		if i > 0 {
			fullText += " "
		}
		fullText += token.Text
	}
	
	// Strong patterns that indicate intensity
	if strings.Contains(strings.ToUpper(fullText), "!!!") {
		features["triple_exclamation"] = 1.0
	}
	if strings.Contains(strings.ToUpper(fullText), "NEVER") ||
	   strings.Contains(strings.ToUpper(fullText), "ALWAYS") {
		features["absolute_language"] = 1.0
	}
	
	// Emotional expression patterns
	if strings.Contains(strings.ToLower(fullText), "can't believe") ||
	   strings.Contains(strings.ToLower(fullText), "cannot believe") {
		features["disbelief_expression"] = 1.0
	}
	
	// Count sentiment word density relative to text length
	if len(docTokens) > 0 {
		sentimentDensity := totalSentiment / float64(len(docTokens))
		features["sentiment_density"] = sentimentDensity
		
		if sentimentDensity > 0.3 {
			features["high_sentiment_density"] = 1.0
		}
	}
	
	return features
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
	if score1.Confidence == 0 || len(score1.Scores) == 0 {
		return score2
	}
	
	// If second score is empty, return first  
	if score2.Confidence == 0 || len(score2.Scores) == 0 {
		return score1
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
	
	// Get all possible classes from both scores
	allClasses := make(map[SentimentClass]bool)
	for class := range score1.Scores {
		allClasses[class] = true
	}
	for class := range score2.Scores {
		allClasses[class] = true
	}
	
	// Combine scores for all classes
	for class := range allClasses {
		prob1 := score1.Scores[class]
		prob2 := score2.Scores[class]
		combined.Scores[class] = prob1*weight1 + prob2*weight2
	}
	
	// Normalize combined probabilities to sum to 1.0
	totalProb := 0.0
	for _, prob := range combined.Scores {
		totalProb += prob
	}
	if totalProb > 0 {
		for class := range combined.Scores {
			combined.Scores[class] /= totalProb
		}
		
		// Recalculate dominant class after normalization
		maxProb := 0.0
		for class, prob := range combined.Scores {
			if prob > maxProb {
				maxProb = prob
				combined.Dominant = class
			}
		}
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
	// If ML model is not available, return neutral score
	if sc.model == nil {
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

	// Convert float64 features to string features for the model
	stringFeatures := make(map[string]string)
	for key, value := range features {
		stringFeatures[key] = fmt.Sprintf("%.6f", value)
	}

	// Calculate scores for each sentiment class
	classScores := make(map[string]float64)
	maxScore := math.Inf(-1)
	
	// Calculate scores for each sentiment class
	for _, label := range sc.model.labels {
		score := 0.0
		encoded := sc.model.encodeSentimentFeatures(stringFeatures, label)
		
		for _, feature := range encoded {
			if feature.key < len(sc.model.weights) {
				score += sc.model.weights[feature.key] * float64(feature.value)
			}
		}
		
		classScores[label] = score
		if score > maxScore {
			maxScore = score
		}
	}
	
	
	// Apply softmax to convert scores to probabilities
	probabilities := make(map[SentimentClass]float64)
	totalExp := 0.0
	
	for label, score := range classScores {
		exp := math.Exp(score - maxScore) // Subtract maxScore for numerical stability
		probabilities[SentimentClass(label)] = exp
		totalExp += exp
	}
	
	// Normalize probabilities
	for class := range probabilities {
		probabilities[class] /= totalExp
	}
	
	// Ensure all 5 sentiment classes are included, even if they have 0 probability
	allSentimentClasses := []SentimentClass{StrongNegative, Negative, Neutral, Positive, StrongPositive}
	for _, class := range allSentimentClasses {
		if _, exists := probabilities[class]; !exists {
			probabilities[class] = 0.0
		}
	}
	
	// Force renormalization of probabilities to ensure they sum to 1.0
	total := 0.0
	for _, prob := range probabilities {
		total += prob
	}
	if total > 0 {
		for class := range probabilities {
			probabilities[class] /= total
		}
	}
	
	// Find dominant class
	dominant := Neutral
	maxProb := 0.0
	for class, prob := range probabilities {
		if prob > maxProb {
			maxProb = prob
			dominant = class
		}
	}
	
	// Calculate polarity and intensity
	polarity := sc.calculatePolarity(probabilities)
	intensity := sc.calculateIntensity(probabilities)
	confidence := maxProb // Confidence is the probability of the dominant class
	
	return SentimentScore{
		Polarity:     polarity,
		Intensity:    intensity,
		Confidence:   confidence,
		Dominant:     dominant,
		Scores:       probabilities,
		Subjectivity: sc.calculateSubjectivity(probabilities),
	}
}

// calculatePolarity calculates sentiment polarity from class probabilities
func (sc *sentimentClassifier) calculatePolarity(probabilities map[SentimentClass]float64) float64 {
	positive := probabilities[StrongPositive]*1.0 + probabilities[Positive]*0.5
	negative := probabilities[StrongNegative]*1.0 + probabilities[Negative]*0.5
	
	return positive - negative
}

// calculateIntensity calculates sentiment intensity from class probabilities
func (sc *sentimentClassifier) calculateIntensity(probabilities map[SentimentClass]float64) float64 {
	strong := probabilities[StrongPositive] + probabilities[StrongNegative]
	mild := probabilities[Positive] + probabilities[Negative]
	
	return strong*1.0 + mild*0.5
}

// calculateSubjectivity calculates subjectivity from class probabilities
func (sc *sentimentClassifier) calculateSubjectivity(probabilities map[SentimentClass]float64) float64 {
	// Subjectivity is 1 - probability of neutral
	return 1.0 - probabilities[Neutral]
}

// loadBasicSentimentModel loads a basic sentiment classification model
func loadBasicSentimentModel(lang Language) *sentimentClassifier {
	// For Phase 1, return basic classifier with feature extractor
	return &sentimentClassifier{
		model:    nil, // Will be implemented in Phase 2
		features: newSentimentFeatureExtractor(lang),
	}
}

// loadSentimentModelFromModel loads a sentiment classifier with trained model
func loadSentimentModelFromModel(lang Language, model *binaryMaxentClassifier) *sentimentClassifier {
	return &sentimentClassifier{
		model:    model,
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