package prose

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
	"unicode"
)

// sentimentFeatureExtractor extracts features for ML classification
type sentimentFeatureExtractor struct {
	ngramSize     int
	usePOS        bool
	useDependency bool
	language      Language
	externalWords map[string]map[string]bool // category -> words map
}

// newSentimentFeatureExtractor creates a new feature extractor
func newSentimentFeatureExtractor(lang Language) *sentimentFeatureExtractor {
	return &sentimentFeatureExtractor{
		ngramSize:     3,
		usePOS:        true,
		useDependency: false, // Will be enabled when dependency parsing is available
		language:      lang,
		externalWords: make(map[string]map[string]bool),
	}
}

// newSentimentFeatureExtractorWithExternal creates a feature extractor with external word lists
func newSentimentFeatureExtractorWithExternal(lang Language, externalPath string) (*sentimentFeatureExtractor, error) {
	extractor := &sentimentFeatureExtractor{
		ngramSize:     3,
		usePOS:        true,
		useDependency: false,
		language:      lang,
		externalWords: make(map[string]map[string]bool),
	}

	if externalPath != "" {
		if err := extractor.loadExternalWords(externalPath); err != nil {
			return nil, err
		}
	}

	return extractor, nil
}

// loadExternalWords loads external word lists for feature extraction
func (sfe *sentimentFeatureExtractor) loadExternalWords(filepath string) error {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return fmt.Errorf("error reading external words file: %w", err)
	}

	var external ExternalLexicon
	if err := json.Unmarshal(data, &external); err != nil {
		return fmt.Errorf("error parsing external words JSON: %w", err)
	}

	langKey := languageToJSONKey(sfe.language)
	if langData, exists := external.Languages[langKey]; exists {
		sfe.loadLanguageWords(langData)
	}

	return nil
}


// loadLanguageWords loads word lists for the specified language
func (sfe *sentimentFeatureExtractor) loadLanguageWords(data LanguageLexicon) {
	// Initialize categories
	categories := []string{
		"positive", "negative", "intensifiers", "diminishers",
		"joy", "anger", "fear", "sadness", "surprise", "negations",
		"modal_verbs", "discourse_markers", "subjective",
	}

	for _, category := range categories {
		if sfe.externalWords[category] == nil {
			sfe.externalWords[category] = make(map[string]bool)
		}
	}

	// Load positive words
	for _, entry := range data.Positive {
		sfe.externalWords["positive"][strings.ToLower(entry.Word)] = true
	}

	// Load negative words
	for _, entry := range data.Negative {
		sfe.externalWords["negative"][strings.ToLower(entry.Word)] = true
	}

	// Load intensifiers
	for _, word := range data.Intensifiers {
		sfe.externalWords["intensifiers"][strings.ToLower(word)] = true
	}

	// Load diminishers
	for _, word := range data.Diminishers {
		sfe.externalWords["diminishers"][strings.ToLower(word)] = true
	}

	// Load negations
	for _, word := range data.Negations {
		sfe.externalWords["negations"][strings.ToLower(word)] = true
	}
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

	// 6. Punctuation and style features
	sfe.extractStyleFeatures(tokens, features)

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

	// Character n-grams for capturing morphological patterns
	sfe.extractCharNGrams(tokens, features)
}

// extractCharNGrams extracts character-level n-grams
func (sfe *sentimentFeatureExtractor) extractCharNGrams(tokens []*Token, features map[string]float64) {
	for _, token := range tokens {
		word := strings.ToLower(token.Text)

		// Skip very short words and punctuation
		if len(word) < 3 || !isWord(word) {
			continue
		}

		// Prefix features (first 2-3 chars)
		if len(word) >= 2 {
			features["prefix2:"+word[:2]] = 1.0
		}
		if len(word) >= 3 {
			features["prefix3:"+word[:3]] = 1.0
		}

		// Suffix features (last 2-3 chars)
		if len(word) >= 2 {
			features["suffix2:"+word[len(word)-2:]] = 1.0
		}
		if len(word) >= 3 {
			features["suffix3:"+word[len(word)-3:]] = 1.0
		}
	}
}

// extractPOSPatterns extracts POS-based patterns
func (sfe *sentimentFeatureExtractor) extractPOSPatterns(tokens []*Token, features map[string]float64) {
	// Count POS tags
	posCounts := make(map[string]int)
	for _, token := range tokens {
		if token.Tag != "" {
			posCounts[token.Tag]++
		}
	}

	// Normalize POS counts
	totalTokens := float64(len(tokens))
	for tag, count := range posCounts {
		features["pos_ratio:"+tag] = float64(count) / totalTokens
	}

	// POS sequences
	for i := 0; i < len(tokens)-1; i++ {
		if tokens[i].Tag != "" && tokens[i+1].Tag != "" {
			pattern := tokens[i].Tag + "_" + tokens[i+1].Tag
			key := "pos_bigram:" + pattern
			features[key] = 1.0
		}
	}

	// Adjective-Noun patterns
	for i := 0; i < len(tokens)-1; i++ {
		if strings.HasPrefix(tokens[i].Tag, "JJ") && strings.HasPrefix(tokens[i+1].Tag, "NN") {
			key := "adj_noun:" + strings.ToLower(tokens[i].Text) + "_" + strings.ToLower(tokens[i+1].Text)
			features[key] = 1.0
			features["has_adj_noun"] = 1.0
		}
	}

	// Verb-Adverb patterns
	for i := 0; i < len(tokens)-1; i++ {
		if strings.HasPrefix(tokens[i].Tag, "VB") && strings.HasPrefix(tokens[i+1].Tag, "RB") {
			key := "verb_adv:" + strings.ToLower(tokens[i].Text) + "_" + strings.ToLower(tokens[i+1].Text)
			features[key] = 1.0
			features["has_verb_adv"] = 1.0
		}
	}

	// Adverb-Adjective patterns (for intensity)
	for i := 0; i < len(tokens)-1; i++ {
		if strings.HasPrefix(tokens[i].Tag, "RB") && strings.HasPrefix(tokens[i+1].Tag, "JJ") {
			key := "adv_adj:" + strings.ToLower(tokens[i].Text) + "_" + strings.ToLower(tokens[i+1].Text)
			features[key] = 1.0
			features["has_adv_adj"] = 1.0
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
		emojiCount       int
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
			features["has_elongated"] = 1.0
		}

		// Emoji detection (basic)
		if containsEmoji(token.Text) {
			emojiCount++
		}
	}

	// Normalize by token count
	tokenCount := float64(len(tokens))
	if tokenCount > 0 {
		features["exclamation_ratio"] = float64(exclamationCount) / tokenCount
		features["question_ratio"] = float64(questionCount) / tokenCount
		features["caps_ratio"] = float64(capsCount) / tokenCount
		features["elongated_ratio"] = float64(elongatedCount) / tokenCount
		features["emoji_ratio"] = float64(emojiCount) / tokenCount
	}

	// Binary features
	if exclamationCount > 0 {
		features["has_exclamation"] = 1.0
	}
	if exclamationCount > 2 {
		features["multiple_exclamations"] = 1.0
	}
	if questionCount > 0 {
		features["has_question"] = 1.0
	}
	if capsCount > 0 {
		features["has_caps"] = 1.0
	}
	if emojiCount > 0 {
		features["has_emoji"] = 1.0
	}

	// Text length features
	features["token_count"] = tokenCount
	features["token_count_log"] = math.Log(tokenCount + 1)
	features["avg_word_length"] = calculateAvgWordLength(tokens)
}

// extractSyntacticFeatures extracts syntax-based features
func (sfe *sentimentFeatureExtractor) extractSyntacticFeatures(tokens []*Token, features map[string]float64) {
	var (
		negationCount    int
		comparativeCount int
		superlativeCount int
		modalCount       int
	)

	for _, token := range tokens {
		lowerText := strings.ToLower(token.Text)

		// Negation detection
		if sfe.isNegationWord(lowerText) {
			negationCount++
		}

		// Comparative and superlative adjectives
		if token.Tag == "JJR" { // Comparative
			comparativeCount++
		}
		if token.Tag == "JJS" { // Superlative
			superlativeCount++
		}

		// Modal verbs
		if token.Tag == "MD" || sfe.isModalVerb(lowerText) {
			modalCount++
		}
	}

	features["negation_count"] = float64(negationCount)
	features["comparative_count"] = float64(comparativeCount)
	features["superlative_count"] = float64(superlativeCount)
	features["modal_count"] = float64(modalCount)

	// Binary features
	if negationCount > 0 {
		features["has_negation"] = 1.0
	}
	if comparativeCount > 0 {
		features["has_comparative"] = 1.0
	}
	if superlativeCount > 0 {
		features["has_superlative"] = 1.0
	}
	if modalCount > 0 {
		features["has_modal"] = 1.0
	}

	// Syntactic complexity
	features["syntactic_complexity"] = calculateSyntacticComplexity(tokens)
}

// extractSemanticFeatures extracts meaning-based features
func (sfe *sentimentFeatureExtractor) extractSemanticFeatures(tokens []*Token, features map[string]float64) {
	// Sentiment word counts (would use lexicon in full implementation)
	var (
		posWordCount     int
		negWordCount     int
		intensifierCount int
		diminisherCount  int
	)

	// Get language-specific indicators
	positiveIndicators := sfe.getPositiveIndicators()
	negativeIndicators := sfe.getNegativeIndicators()
	intensifiers := sfe.getIntensifiers()
	diminishers := sfe.getDiminishers()

	for _, token := range tokens {
		lowerText := strings.ToLower(token.Text)

		if positiveIndicators[lowerText] {
			posWordCount++
		}
		if negativeIndicators[lowerText] {
			negWordCount++
		}
		if intensifiers[lowerText] {
			intensifierCount++
		}
		if diminishers[lowerText] {
			diminisherCount++
		}
	}

	tokenCount := float64(len(tokens))
	if tokenCount > 0 {
		features["pos_word_ratio"] = float64(posWordCount) / tokenCount
		features["neg_word_ratio"] = float64(negWordCount) / tokenCount
		features["intensifier_ratio"] = float64(intensifierCount) / tokenCount
		features["diminisher_ratio"] = float64(diminisherCount) / tokenCount
	}

	// Binary semantic features
	if posWordCount > 0 {
		features["has_positive"] = 1.0
	}
	if negWordCount > 0 {
		features["has_negative"] = 1.0
	}
	if posWordCount > 0 && negWordCount > 0 {
		features["has_mixed"] = 1.0
	}
	if intensifierCount > 0 {
		features["has_intensifier"] = 1.0
	}
	if diminisherCount > 0 {
		features["has_diminisher"] = 1.0
	}

	// Emotion categories
	sfe.extractEmotionFeatures(tokens, features)
}

// extractEmotionFeatures adds emotion-specific features
func (sfe *sentimentFeatureExtractor) extractEmotionFeatures(tokens []*Token, features map[string]float64) {
	// Get language-specific emotion indicators
	joyWords := sfe.getJoyWords()
	angerWords := sfe.getAngerWords()
	fearWords := sfe.getFearWords()
	sadnessWords := sfe.getSadnessWords()
	surpriseWords := sfe.getSurpriseWords()

	for _, token := range tokens {
		lowerText := strings.ToLower(token.Text)

		if joyWords[lowerText] {
			features["has_joy_words"] = 1.0
		}
		if angerWords[lowerText] {
			features["has_anger_words"] = 1.0
		}
		if fearWords[lowerText] {
			features["has_fear_words"] = 1.0
		}
		if sadnessWords[lowerText] {
			features["has_sadness_words"] = 1.0
		}
		if surpriseWords[lowerText] {
			features["has_surprise_words"] = 1.0
		}
	}
}

// extractStyleFeatures extracts writing style features
func (sfe *sentimentFeatureExtractor) extractStyleFeatures(tokens []*Token, features map[string]float64) {
	// Count different punctuation types
	punctCounts := make(map[string]int)
	for _, token := range tokens {
		if isPunctuation(token.Text) {
			punctCounts[token.Text]++
		}
	}

	// Punctuation ratios
	tokenCount := float64(len(tokens))
	if tokenCount > 0 {
		for punct, count := range punctCounts {
			key := "punct_" + punct + "_ratio"
			features[key] = float64(count) / tokenCount
		}
	}

	// Discourse markers
	discourseMarkers := sfe.getDiscourseMarkers()

	discourseCount := 0
	for _, token := range tokens {
		if discourseMarkers[strings.ToLower(token.Text)] {
			discourseCount++
			features["has_discourse_marker"] = 1.0
		}
	}
	features["discourse_marker_count"] = float64(discourseCount)

	// Subjectivity indicators
	subjectiveWords := sfe.getSubjectiveWords()

	subjectiveCount := 0
	for _, token := range tokens {
		if subjectiveWords[strings.ToLower(token.Text)] {
			subjectiveCount++
			features["has_subjective"] = 1.0
		}
	}
	features["subjective_count"] = float64(subjectiveCount)
}

// Helper functions

func isAllCaps(text string) bool {
	if len(text) == 0 {
		return false
	}
	hasLetter := false
	for _, r := range text {
		if unicode.IsLetter(r) {
			hasLetter = true
			if !unicode.IsUpper(r) {
				return false
			}
		}
	}
	return hasLetter
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

func isPunctuation(text string) bool {
	if len(text) == 0 {
		return false
	}
	for _, r := range text {
		if !unicode.IsPunct(r) && !unicode.IsSymbol(r) {
			return false
		}
	}
	return true
}

func containsEmoji(text string) bool {
	// Basic emoji detection for common ranges
	for _, r := range text {
		if r >= 0x1F600 && r <= 0x1F64F || // Emoticons
			r >= 0x1F300 && r <= 0x1F5FF || // Misc Symbols and Pictographs
			r >= 0x1F680 && r <= 0x1F6FF || // Transport and Map
			r >= 0x2600 && r <= 0x26FF || // Misc symbols
			r >= 0x2700 && r <= 0x27BF { // Dingbats
			return true
		}
	}
	return false
}

func (sfe *sentimentFeatureExtractor) isNegationWord(word string) bool {
	negations := sfe.getNegationWords()
	return negations[word]
}

func (sfe *sentimentFeatureExtractor) isModalVerb(word string) bool {
	modals := sfe.getModalVerbs()
	return modals[word]
}

func calculateSyntacticComplexity(tokens []*Token) float64 {
	if len(tokens) == 0 {
		return 0
	}

	// Simple complexity based on token diversity and sentence structure
	uniqueWords := make(map[string]bool)
	punctCount := 0
	conjunctionCount := 0

	conjunctions := map[string]bool{
		"and": true, "or": true, "but": true, "because": true,
		"although": true, "while": true, "since": true, "unless": true,
		"if": true, "when": true, "where": true, "whereas": true,
	}

	for _, token := range tokens {
		if isWord(token.Text) {
			uniqueWords[strings.ToLower(token.Text)] = true
		}
		if isPunctuation(token.Text) {
			punctCount++
		}
		if conjunctions[strings.ToLower(token.Text)] {
			conjunctionCount++
		}
	}

	// Type-token ratio
	ttr := float64(len(uniqueWords)) / float64(len(tokens))

	// Punctuation complexity
	punctRatio := float64(punctCount) / float64(len(tokens))

	// Conjunction complexity
	conjRatio := float64(conjunctionCount) / float64(len(tokens))

	// Combined complexity score
	complexity := (ttr * 0.5) + (punctRatio * 0.3) + (conjRatio * 0.2)

	return math.Min(1.0, complexity)
}

// Language-specific word lists

// getPositiveIndicators returns positive sentiment words for the language
func (sfe *sentimentFeatureExtractor) getPositiveIndicators() map[string]bool {
	// Start with hardcoded core words
	result := sfe.getCorePositiveIndicators()
	
	// Add external words if available
	if externalWords, exists := sfe.externalWords["positive"]; exists {
		for word := range externalWords {
			result[word] = true
		}
	}
	
	return result
}

// getCorePositiveIndicators returns core hardcoded positive words
func (sfe *sentimentFeatureExtractor) getCorePositiveIndicators() map[string]bool {
	switch sfe.language {
	case English:
		return map[string]bool{
			"good": true, "great": true, "excellent": true, "love": true,
			"best": true, "happy": true, "wonderful": true, "amazing": true,
			"perfect": true, "beautiful": true, "fantastic": true, "awesome": true,
		}
	case Spanish:
		return map[string]bool{
			"bueno": true, "excelente": true, "maravilloso": true, "fantástico": true,
			"mejor": true, "feliz": true, "hermoso": true, "perfecto": true,
			"amor": true, "genial": true, "increíble": true, "estupendo": true,
		}
	case French:
		return map[string]bool{
			"bon": true, "excellent": true, "merveilleux": true, "fantastique": true,
			"meilleur": true, "heureux": true, "beau": true, "parfait": true,
			"amour": true, "génial": true, "incroyable": true, "magnifique": true,
		}
	case German:
		return map[string]bool{
			"gut": true, "ausgezeichnet": true, "wunderbar": true, "fantastisch": true,
			"besser": true, "glücklich": true, "schön": true, "perfekt": true,
			"liebe": true, "großartig": true, "unglaublich": true, "herrlich": true,
		}
	case Japanese:
		return map[string]bool{
			"良い": true, "いい": true, "素晴らしい": true, "すごい": true,
			"大好き": true, "嬉しい": true, "美しい": true, "完璧": true,
			"最高": true, "楽しい": true, "優秀": true, "立派": true,
		}
	default:
		// Default to English
		return map[string]bool{
			"good": true, "great": true, "excellent": true, "love": true,
			"best": true, "happy": true, "wonderful": true, "amazing": true,
			"perfect": true, "beautiful": true, "fantastic": true, "awesome": true,
		}
	}
}

// getNegativeIndicators returns negative sentiment words for the language
func (sfe *sentimentFeatureExtractor) getNegativeIndicators() map[string]bool {
	// Start with hardcoded core words
	result := sfe.getCoreNegativeIndicators()
	
	// Add external words if available
	if externalWords, exists := sfe.externalWords["negative"]; exists {
		for word := range externalWords {
			result[word] = true
		}
	}
	
	return result
}

// getCoreNegativeIndicators returns core hardcoded negative words
func (sfe *sentimentFeatureExtractor) getCoreNegativeIndicators() map[string]bool {
	switch sfe.language {
	case English:
		return map[string]bool{
			"bad": true, "terrible": true, "awful": true, "hate": true,
			"worst": true, "sad": true, "horrible": true, "disgusting": true,
			"poor": true, "disappointing": true, "fail": true, "wrong": true,
		}
	case Spanish:
		return map[string]bool{
			"malo": true, "terrible": true, "horrible": true, "odio": true,
			"peor": true, "triste": true, "feo": true, "decepcionante": true,
			"pobre": true, "fallar": true, "mal": true, "disgusto": true,
		}
	case French:
		return map[string]bool{
			"mauvais": true, "terrible": true, "horrible": true, "déteste": true,
			"pire": true, "triste": true, "laid": true, "décevant": true,
			"pauvre": true, "échouer": true, "mal": true, "dégoûtant": true,
		}
	case German:
		return map[string]bool{
			"schlecht": true, "schrecklich": true, "furchtbar": true, "hasse": true,
			"schlechter": true, "traurig": true, "hässlich": true, "enttäuschend": true,
			"arm": true, "versagen": true, "falsch": true, "ekelhaft": true,
		}
	case Japanese:
		return map[string]bool{
			"悪い": true, "ひどい": true, "嫌い": true, "悲しい": true,
			"つまらない": true, "最悪": true, "残念": true, "怖い": true,
			"嫌": true, "失敗": true, "間違い": true, "不快": true,
		}
	default:
		// Default to English
		return map[string]bool{
			"bad": true, "terrible": true, "awful": true, "hate": true,
			"worst": true, "sad": true, "horrible": true, "disgusting": true,
			"poor": true, "disappointing": true, "fail": true, "wrong": true,
		}
	}
}

// getIntensifiers returns intensifying words for the language
func (sfe *sentimentFeatureExtractor) getIntensifiers() map[string]bool {
	// Start with hardcoded core words
	result := sfe.getCoreIntensifiers()
	
	// Add external words if available
	if externalWords, exists := sfe.externalWords["intensifiers"]; exists {
		for word := range externalWords {
			result[word] = true
		}
	}
	
	return result
}

// getCoreIntensifiers returns core hardcoded intensifiers
func (sfe *sentimentFeatureExtractor) getCoreIntensifiers() map[string]bool {
	switch sfe.language {
	case English:
		return map[string]bool{
			"very": true, "extremely": true, "absolutely": true, "totally": true,
			"really": true, "so": true, "quite": true, "incredibly": true,
		}
	case Spanish:
		return map[string]bool{
			"muy": true, "extremadamente": true, "absolutamente": true, "totalmente": true,
			"realmente": true, "tan": true, "bastante": true, "increíblemente": true,
		}
	case French:
		return map[string]bool{
			"très": true, "extrêmement": true, "absolument": true, "totalement": true,
			"vraiment": true, "si": true, "assez": true, "incroyablement": true,
		}
	case German:
		return map[string]bool{
			"sehr": true, "extrem": true, "absolut": true, "total": true,
			"wirklich": true, "so": true, "ziemlich": true, "unglaublich": true,
		}
	case Japanese:
		return map[string]bool{
			"とても": true, "すごく": true, "非常に": true, "本当に": true,
			"かなり": true, "めちゃくちゃ": true, "超": true, "完全に": true,
		}
	default:
		// Default to English
		return map[string]bool{
			"very": true, "extremely": true, "absolutely": true, "totally": true,
			"really": true, "so": true, "quite": true, "incredibly": true,
		}
	}
}

// getDiminishers returns diminishing words for the language
func (sfe *sentimentFeatureExtractor) getDiminishers() map[string]bool {
	// Start with hardcoded core words
	result := sfe.getCoreDiminishers()
	
	// Add external words if available
	if externalWords, exists := sfe.externalWords["diminishers"]; exists {
		for word := range externalWords {
			result[word] = true
		}
	}
	
	return result
}

// getCoreDiminishers returns core hardcoded diminishers
func (sfe *sentimentFeatureExtractor) getCoreDiminishers() map[string]bool {
	switch sfe.language {
	case English:
		return map[string]bool{
			"slightly": true, "somewhat": true, "rather": true, "fairly": true,
			"barely": true, "hardly": true, "scarcely": true, "marginally": true,
		}
	case Spanish:
		return map[string]bool{
			"ligeramente": true, "algo": true, "bastante": true, "apenas": true,
			"casi": true, "poco": true, "escasamente": true, "marginalmente": true,
		}
	case French:
		return map[string]bool{
			"légèrement": true, "quelque peu": true, "plutôt": true, "assez": true,
			"à peine": true, "presque": true, "peu": true, "marginalement": true,
		}
	case German:
		return map[string]bool{
			"leicht": true, "etwas": true, "ziemlich": true, "kaum": true,
			"fast": true, "wenig": true, "knapp": true, "marginal": true,
		}
	case Japanese:
		return map[string]bool{
			"少し": true, "ちょっと": true, "やや": true, "わずかに": true,
			"あまり": true, "そんなに": true, "それほど": true, "たいして": true,
		}
	default:
		// Default to English
		return map[string]bool{
			"slightly": true, "somewhat": true, "rather": true, "fairly": true,
			"barely": true, "hardly": true, "scarcely": true, "marginally": true,
		}
	}
}

// getJoyWords returns joy emotion words for the language
func (sfe *sentimentFeatureExtractor) getJoyWords() map[string]bool {
	switch sfe.language {
	case English:
		return map[string]bool{
			"happy": true, "joy": true, "cheerful": true, "delighted": true,
			"pleased": true, "glad": true, "joyful": true, "elated": true,
		}
	case Spanish:
		return map[string]bool{
			"feliz": true, "alegría": true, "alegre": true, "encantado": true,
			"contento": true, "gozoso": true, "jubiloso": true, "eufórico": true,
		}
	case French:
		return map[string]bool{
			"heureux": true, "joie": true, "joyeux": true, "ravi": true,
			"content": true, "gai": true, "réjoui": true, "euphorique": true,
		}
	case German:
		return map[string]bool{
			"glücklich": true, "freude": true, "fröhlich": true, "erfreut": true,
			"zufrieden": true, "froh": true, "freudig": true, "euphorisch": true,
		}
	case Japanese:
		return map[string]bool{
			"嬉しい": true, "楽しい": true, "喜び": true, "幸せ": true,
			"満足": true, "興奮": true, "陽気": true, "明るい": true,
		}
	default:
		// Default to English
		return map[string]bool{
			"happy": true, "joy": true, "cheerful": true, "delighted": true,
			"pleased": true, "glad": true, "joyful": true, "elated": true,
		}
	}
}

// getAngerWords returns anger emotion words for the language
func (sfe *sentimentFeatureExtractor) getAngerWords() map[string]bool {
	switch sfe.language {
	case English:
		return map[string]bool{
			"angry": true, "mad": true, "furious": true, "rage": true,
			"annoyed": true, "irritated": true, "frustrated": true, "upset": true,
		}
	case Spanish:
		return map[string]bool{
			"enojado": true, "furioso": true, "rabia": true, "ira": true,
			"molesto": true, "irritado": true, "frustrado": true, "enfadado": true,
		}
	case French:
		return map[string]bool{
			"en colère": true, "furieux": true, "rage": true, "colère": true,
			"agacé": true, "irrité": true, "frustré": true, "contrarié": true,
		}
	case German:
		return map[string]bool{
			"wütend": true, "verrückt": true, "böse": true, "zorn": true,
			"verärgert": true, "gereizt": true, "frustriert": true, "aufgebracht": true,
		}
	case Japanese:
		return map[string]bool{
			"怒り": true, "腹立つ": true, "イライラ": true, "ムカつく": true,
			"憤慨": true, "激怒": true, "不満": true, "苛立ち": true,
		}
	default:
		// Default to English
		return map[string]bool{
			"angry": true, "mad": true, "furious": true, "rage": true,
			"annoyed": true, "irritated": true, "frustrated": true, "upset": true,
		}
	}
}

// getFearWords returns fear emotion words for the language
func (sfe *sentimentFeatureExtractor) getFearWords() map[string]bool {
	switch sfe.language {
	case English:
		return map[string]bool{
			"afraid": true, "scared": true, "frightened": true, "terrified": true,
			"anxious": true, "nervous": true, "worried": true, "panic": true,
		}
	case Spanish:
		return map[string]bool{
			"miedo": true, "asustado": true, "aterrorizado": true, "ansioso": true,
			"nervioso": true, "preocupado": true, "pánico": true, "temor": true,
		}
	case French:
		return map[string]bool{
			"peur": true, "effrayé": true, "terrifié": true, "anxieux": true,
			"nerveux": true, "inquiet": true, "panique": true, "crainte": true,
		}
	case German:
		return map[string]bool{
			"angst": true, "erschrocken": true, "verängstigt": true, "ängstlich": true,
			"nervös": true, "besorgt": true, "panik": true, "furcht": true,
		}
	case Japanese:
		return map[string]bool{
			"怖い": true, "恐怖": true, "不安": true, "心配": true,
			"緊張": true, "恐れ": true, "びくびく": true, "驚く": true,
		}
	default:
		// Default to English
		return map[string]bool{
			"afraid": true, "scared": true, "frightened": true, "terrified": true,
			"anxious": true, "nervous": true, "worried": true, "panic": true,
		}
	}
}

// getSadnessWords returns sadness emotion words for the language
func (sfe *sentimentFeatureExtractor) getSadnessWords() map[string]bool {
	switch sfe.language {
	case English:
		return map[string]bool{
			"sad": true, "depressed": true, "miserable": true, "gloomy": true,
			"unhappy": true, "sorrowful": true, "melancholy": true, "dejected": true,
		}
	case Spanish:
		return map[string]bool{
			"triste": true, "deprimido": true, "miserable": true, "sombrío": true,
			"infeliz": true, "doliente": true, "melancólico": true, "abatido": true,
		}
	case French:
		return map[string]bool{
			"triste": true, "déprimé": true, "misérable": true, "sombre": true,
			"malheureux": true, "affligé": true, "mélancolique": true, "abattu": true,
		}
	case German:
		return map[string]bool{
			"traurig": true, "deprimiert": true, "elend": true, "düster": true,
			"unglücklich": true, "betrübt": true, "melancholisch": true, "niedergeschlagen": true,
		}
	case Japanese:
		return map[string]bool{
			"悲しい": true, "憂鬱": true, "落ち込む": true, "淋しい": true,
			"寂しい": true, "悲哀": true, "失望": true, "絶望": true,
		}
	default:
		// Default to English
		return map[string]bool{
			"sad": true, "depressed": true, "miserable": true, "gloomy": true,
			"unhappy": true, "sorrowful": true, "melancholy": true, "dejected": true,
		}
	}
}

// getSurpriseWords returns surprise emotion words for the language
func (sfe *sentimentFeatureExtractor) getSurpriseWords() map[string]bool {
	switch sfe.language {
	case English:
		return map[string]bool{
			"surprised": true, "amazed": true, "astonished": true, "shocked": true,
			"stunned": true, "startled": true, "unexpected": true, "sudden": true,
		}
	case Spanish:
		return map[string]bool{
			"sorprendido": true, "asombrado": true, "atónito": true, "conmocionado": true,
			"aturdido": true, "sobresaltado": true, "inesperado": true, "repentino": true,
		}
	case French:
		return map[string]bool{
			"surpris": true, "étonné": true, "stupéfait": true, "choqué": true,
			"abasourdi": true, "sursauté": true, "inattendu": true, "soudain": true,
		}
	case German:
		return map[string]bool{
			"überrascht": true, "erstaunt": true, "verblüfft": true, "schockiert": true,
			"betäubt": true, "erschrocken": true, "unerwartet": true, "plötzlich": true,
		}
	case Japanese:
		return map[string]bool{
			"驚く": true, "びっくり": true, "衝撃": true, "意外": true,
			"突然": true, "予想外": true, "まさか": true, "驚き": true,
		}
	default:
		// Default to English
		return map[string]bool{
			"surprised": true, "amazed": true, "astonished": true, "shocked": true,
			"stunned": true, "startled": true, "unexpected": true, "sudden": true,
		}
	}
}

// getNegationWords returns negation words for the language
func (sfe *sentimentFeatureExtractor) getNegationWords() map[string]bool {
	// Start with hardcoded core words
	result := sfe.getCoreNegationWords()
	
	// Add external words if available
	if externalWords, exists := sfe.externalWords["negations"]; exists {
		for word := range externalWords {
			result[word] = true
		}
	}
	
	return result
}

// getCoreNegationWords returns core hardcoded negation words
func (sfe *sentimentFeatureExtractor) getCoreNegationWords() map[string]bool {
	switch sfe.language {
	case English:
		return map[string]bool{
			"not": true, "no": true, "never": true, "neither": true,
			"nor": true, "cannot": true, "can't": true, "won't": true,
			"don't": true, "doesn't": true, "didn't": true, "isn't": true,
			"aren't": true, "wasn't": true, "weren't": true, "without": true,
		}
	case Spanish:
		return map[string]bool{
			"no": true, "nunca": true, "jamás": true, "ni": true,
			"sin": true, "nada": true, "nadie": true, "ningún": true,
			"ninguna": true, "tampoco": true,
		}
	case French:
		return map[string]bool{
			"ne": true, "pas": true, "non": true, "jamais": true,
			"rien": true, "personne": true, "aucun": true, "aucune": true,
			"ni": true, "sans": true,
		}
	case German:
		return map[string]bool{
			"nicht": true, "nein": true, "kein": true, "keine": true,
			"niemals": true, "nie": true, "nichts": true, "niemand": true,
			"nirgends": true, "ohne": true,
		}
	case Japanese:
		return map[string]bool{
			"ない": true, "いない": true, "ではない": true, "じゃない": true,
			"しない": true, "できない": true, "わからない": true, "だめ": true,
			"いけない": true, "なし": true, "決して": true, "全然": true,
		}
	default:
		// Default to English
		return map[string]bool{
			"not": true, "no": true, "never": true, "neither": true,
			"nor": true, "cannot": true, "can't": true, "won't": true,
			"don't": true, "doesn't": true, "didn't": true, "isn't": true,
			"aren't": true, "wasn't": true, "weren't": true, "without": true,
		}
	}
}

// getModalVerbs returns modal verbs for the language
func (sfe *sentimentFeatureExtractor) getModalVerbs() map[string]bool {
	switch sfe.language {
	case English:
		return map[string]bool{
			"can": true, "could": true, "may": true, "might": true,
			"must": true, "shall": true, "should": true, "will": true,
			"would": true, "ought": true,
		}
	case Spanish:
		return map[string]bool{
			"poder": true, "podría": true, "puede": true, "puedo": true,
			"deber": true, "debería": true, "debe": true, "querer": true,
			"quisiera": true, "querría": true,
		}
	case French:
		return map[string]bool{
			"pouvoir": true, "pourrait": true, "peut": true, "peux": true,
			"devoir": true, "devrait": true, "doit": true, "vouloir": true,
			"voudrait": true, "veut": true,
		}
	case German:
		return map[string]bool{
			"können": true, "könnte": true, "kann": true, "mag": true,
			"müssen": true, "sollte": true, "soll": true, "wollen": true,
			"würde": true, "will": true,
		}
	case Japanese:
		return map[string]bool{
			"できる": true, "かもしれない": true, "だろう": true, "でしょう": true,
			"はず": true, "べき": true, "たい": true, "欲しい": true,
		}
	default:
		// Default to English
		return map[string]bool{
			"can": true, "could": true, "may": true, "might": true,
			"must": true, "shall": true, "should": true, "will": true,
			"would": true, "ought": true,
		}
	}
}

// getDiscourseMarkers returns discourse markers for the language
func (sfe *sentimentFeatureExtractor) getDiscourseMarkers() map[string]bool {
	switch sfe.language {
	case English:
		return map[string]bool{
			"however": true, "therefore": true, "moreover": true, "furthermore": true,
			"although": true, "nevertheless": true, "consequently": true, "meanwhile": true,
			"indeed": true, "actually": true, "basically": true, "obviously": true,
		}
	case Spanish:
		return map[string]bool{
			"sin embargo": true, "por lo tanto": true, "además": true, "asimismo": true,
			"aunque": true, "no obstante": true, "consecuentemente": true, "mientras tanto": true,
			"de hecho": true, "realmente": true, "básicamente": true, "obviamente": true,
		}
	case French:
		return map[string]bool{
			"cependant": true, "par conséquent": true, "de plus": true, "en outre": true,
			"bien que": true, "néanmoins": true, "donc": true, "pendant ce temps": true,
			"en effet": true, "actuellement": true, "fondamentalement": true, "évidemment": true,
		}
	case German:
		return map[string]bool{
			"jedoch": true, "deshalb": true, "außerdem": true, "darüber hinaus": true,
			"obwohl": true, "dennoch": true, "folglich": true, "inzwischen": true,
			"tatsächlich": true, "eigentlich": true, "grundsätzlich": true, "offensichtlich": true,
		}
	case Japanese:
		return map[string]bool{
			"しかし": true, "だから": true, "それで": true, "また": true,
			"でも": true, "ところが": true, "つまり": true, "実際": true,
		}
	default:
		// Default to English
		return map[string]bool{
			"however": true, "therefore": true, "moreover": true, "furthermore": true,
			"although": true, "nevertheless": true, "consequently": true, "meanwhile": true,
			"indeed": true, "actually": true, "basically": true, "obviously": true,
		}
	}
}

// getSubjectiveWords returns subjectivity indicators for the language
func (sfe *sentimentFeatureExtractor) getSubjectiveWords() map[string]bool {
	switch sfe.language {
	case English:
		return map[string]bool{
			"think": true, "believe": true, "feel": true, "seems": true,
			"appears": true, "maybe": true, "perhaps": true, "probably": true,
			"possibly": true, "might": true, "could": true, "would": true,
		}
	case Spanish:
		return map[string]bool{
			"creo": true, "pienso": true, "siento": true, "parece": true,
			"aparece": true, "quizás": true, "tal vez": true, "probablemente": true,
			"posiblemente": true, "podría": true, "puede": true, "sería": true,
		}
	case French:
		return map[string]bool{
			"pense": true, "crois": true, "sens": true, "semble": true,
			"paraît": true, "peut-être": true, "probablement": true, "possiblement": true,
			"pourrait": true, "peut": true, "serait": true, "semblerait": true,
		}
	case German:
		return map[string]bool{
			"denke": true, "glaube": true, "fühle": true, "scheint": true,
			"erscheint": true, "vielleicht": true, "wahrscheinlich": true, "möglicherweise": true,
			"könnte": true, "kann": true, "würde": true, "vermutlich": true,
		}
	case Japanese:
		return map[string]bool{
			"思う": true, "考える": true, "感じる": true, "ようだ": true,
			"みたい": true, "たぶん": true, "おそらく": true, "かもしれない": true,
		}
	default:
		// Default to English
		return map[string]bool{
			"think": true, "believe": true, "feel": true, "seems": true,
			"appears": true, "maybe": true, "perhaps": true, "probably": true,
			"possibly": true, "might": true, "could": true, "would": true,
		}
	}
}
