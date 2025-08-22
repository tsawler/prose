package prose

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"

	"github.com/bbalet/stopwords"
)

// LanguageDetector provides language detection capabilities
type LanguageDetector struct {
	patterns map[Language]*regexp.Regexp
	ngrams   map[Language]map[string]float64
}

// NewLanguageDetector creates a new language detector
func NewLanguageDetector() *LanguageDetector {
	ld := &LanguageDetector{
		patterns: make(map[Language]*regexp.Regexp),
		ngrams:   make(map[Language]map[string]float64),
	}
	
	ld.initializePatterns()
	ld.initializeNgrams()
	
	return ld
}

// initializePatterns sets up language-specific regex patterns
func (ld *LanguageDetector) initializePatterns() {
	// English patterns
	ld.patterns[English] = regexp.MustCompile(`\b(the|and|that|have|for|not|with|you|this|but|his|from|they)\b`)
	
	// Spanish patterns  
	ld.patterns[Spanish] = regexp.MustCompile(`\b(que|de|no|la|el|es|en|un|por|con|como|para|todo|pero|más)\b`)
	
	// French patterns
	ld.patterns[French] = regexp.MustCompile(`\b(de|le|et|à|un|il|être|et|en|avoir|que|pour|dans|ce|son|une)\b`)
	
	// German patterns
	ld.patterns[German] = regexp.MustCompile(`\b(der|die|und|in|den|von|zu|das|mit|sich|des|auf|für|ist|im|dem)\b`)
}

// initializeNgrams sets up language-specific n-gram frequencies
func (ld *LanguageDetector) initializeNgrams() {
	// English trigrams
	ld.ngrams[English] = map[string]float64{
		"the": 0.15, "and": 0.08, "ing": 0.06, "ion": 0.05, "tio": 0.04,
		"ent": 0.03, "ati": 0.03, "for": 0.03, "her": 0.03, "ter": 0.03,
	}
	
	// Spanish trigrams
	ld.ngrams[Spanish] = map[string]float64{
		"que": 0.12, "ión": 0.08, "ado": 0.06, "con": 0.05, "ent": 0.04,
		"par": 0.04, "est": 0.04, "ara": 0.03, "del": 0.03, "los": 0.03,
	}
	
	// French trigrams
	ld.ngrams[French] = map[string]float64{
		"les": 0.10, "ent": 0.08, "ion": 0.07, "des": 0.06, "que": 0.05,
		"ait": 0.04, "lle": 0.04, "eur": 0.04, "our": 0.03, "ant": 0.03,
	}
	
	// German trigrams
	ld.ngrams[German] = map[string]float64{
		"der": 0.12, "und": 0.08, "die": 0.07, "ung": 0.06, "ich": 0.05,
		"ein": 0.04, "sch": 0.04, "den": 0.04, "cht": 0.03, "das": 0.03,
	}
}

// DetectLanguage attempts to detect the language of the given text
func (ld *LanguageDetector) DetectLanguage(text string) (Language, float64) {
	if len(text) < 10 {
		return English, 0.5 // Default to English for short texts
	}
	
	text = strings.ToLower(text)
	scores := make(map[Language]float64)
	
	// Score based on patterns
	for lang, pattern := range ld.patterns {
		matches := pattern.FindAllString(text, -1)
		scores[lang] += float64(len(matches)) * 0.1
	}
	
	// Score based on n-grams
	trigrams := ld.extractTrigrams(text)
	for lang, ngramFreqs := range ld.ngrams {
		for trigram, freq := range trigrams {
			if expectedFreq, exists := ngramFreqs[trigram]; exists {
				scores[lang] += freq * expectedFreq
			}
		}
	}
	
	// Score based on character frequency
	charScores := ld.scoreByCharacterFrequency(text)
	for lang, score := range charScores {
		scores[lang] += score
	}
	
	// Find the language with the highest score
	bestLang := English
	bestScore := 0.0
	totalScore := 0.0
	
	for lang, score := range scores {
		totalScore += score
		if score > bestScore {
			bestScore = score
			bestLang = lang
		}
	}
	
	confidence := bestScore / (totalScore + 1e-10) // Avoid division by zero
	if confidence > 1.0 {
		confidence = 1.0
	}
	
	return bestLang, confidence
}

// extractTrigrams extracts trigrams from text with their frequencies
func (ld *LanguageDetector) extractTrigrams(text string) map[string]float64 {
	trigrams := make(map[string]float64)
	totalTrigrams := 0
	
	runes := []rune(text)
	for i := 0; i <= len(runes)-3; i++ {
		trigram := string(runes[i : i+3])
		if ld.isValidTrigram(trigram) {
			trigrams[trigram]++
			totalTrigrams++
		}
	}
	
	// Normalize frequencies
	for trigram := range trigrams {
		trigrams[trigram] /= float64(totalTrigrams)
	}
	
	return trigrams
}

// isValidTrigram checks if a trigram contains valid characters
func (ld *LanguageDetector) isValidTrigram(trigram string) bool {
	for _, r := range trigram {
		if !unicode.IsLetter(r) {
			return false
		}
	}
	return true
}

// scoreByCharacterFrequency scores languages based on character frequency patterns
func (ld *LanguageDetector) scoreByCharacterFrequency(text string) map[Language]float64 {
	scores := make(map[Language]float64)
	
	// Count character frequencies
	charCount := make(map[rune]int)
	totalChars := 0
	
	for _, r := range text {
		if unicode.IsLetter(r) {
			charCount[r]++
			totalChars++
		}
	}
	
	// Score based on language-specific character patterns
	for char, count := range charCount {
		freq := float64(count) / float64(totalChars)
		
		switch char {
		case 'ñ':
			scores[Spanish] += freq * 10
		case 'ç':
			scores[French] += freq * 8
		case 'ü', 'ö', 'ä', 'ß':
			scores[German] += freq * 8
		case 'q':
			if charCount['u'] > 0 {
				scores[English] += freq * 2
				scores[French] += freq * 2
			}
		case 'w':
			scores[English] += freq * 3
			scores[German] += freq * 2
		case 'k':
			scores[German] += freq * 2
			scores[English] += freq * 1
		case 'j':
			scores[Spanish] += freq * 2
			scores[German] += freq * 1
		}
	}
	
	return scores
}

// LanguageSpecificProcessor provides language-specific processing
type LanguageSpecificProcessor struct {
	language Language
}

// NewLanguageSpecificProcessor creates a processor for a specific language
func NewLanguageSpecificProcessor(lang Language) *LanguageSpecificProcessor {
	return &LanguageSpecificProcessor{language: lang}
}

// GetStopWords returns stop words for the language
func (lsp *LanguageSpecificProcessor) GetStopWords() []string {
	// The stopwords library uses ISO 639-1 language codes
	langCode := string(lsp.language)
	
	// Since the library doesn't export stop words directly, we'll use it functionally
	// by testing words to see if they're removed as stop words
	return getStopWordsForLanguage(langCode)
}

// getStopWordsForLanguage retrieves stop words for a given language code
// Since the bbalet/stopwords library doesn't export stop words directly,
// we detect them by testing if common words get filtered out
func getStopWordsForLanguage(langCode string) []string {
	// Comprehensive list of common stop words across languages
	// We test each to see if the library considers it a stop word
	testWords := getCommonWordsForTesting(langCode)
	
	var stopWords []string
	for _, word := range testWords {
		// Test each word individually to see if it's filtered as a stop word
		cleaned := stopwords.CleanString(word, langCode, false)
		if cleaned == "" || cleaned != word {
			// Word was removed or modified, it's a stop word
			stopWords = append(stopWords, word)
		}
	}
	
	return stopWords
}

// getCommonWordsForTesting returns a comprehensive list of potential stop words to test
func getCommonWordsForTesting(langCode string) []string {
	// Start with common English words that might be stop words
	words := []string{
		// Articles, pronouns, prepositions, conjunctions
		"a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
		"has", "had", "have", "he", "her", "his", "how", "i", "in", "is", "it", 
		"its", "of", "on", "or", "she", "that", "the", "their", "them", "they",
		"this", "to", "was", "we", "were", "what", "when", "where", "which", "who",
		"will", "with", "would", "you", "your",
		// Common verbs and other frequent words
		"about", "after", "all", "also", "am", "any", "back", "because", "before",
		"being", "between", "both", "but", "can", "could", "did", "do", "does",
		"down", "each", "even", "first", "get", "give", "go", "going", "good",
		"got", "had", "has", "have", "here", "him", "himself", "if", "into",
		"just", "know", "last", "like", "made", "make", "many", "may", "me",
		"might", "more", "most", "much", "must", "my", "never", "new", "no",
		"not", "now", "off", "old", "only", "other", "our", "out", "over", 
		"own", "said", "same", "see", "should", "since", "so", "some", "still",
		"such", "take", "than", "then", "there", "these", "thing", "think",
		"those", "through", "time", "too", "two", "under", "up", "upon", "us",
		"use", "used", "using", "very", "want", "way", "well", "went", "were",
		"what", "while", "why", "will", "work", "year", "years", "yet",
	}
	
	// Add language-specific common words based on the language code
	switch langCode {
	case "es": // Spanish
		words = append(words, []string{
			"el", "la", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero",
			"que", "de", "en", "a", "por", "para", "con", "sin", "sobre", "entre",
			"hacia", "hasta", "desde", "durante", "mediante", "ante", "bajo", "contra",
			"según", "tras", "es", "está", "son", "están", "ser", "estar", "hay",
			"había", "fue", "era", "sido", "siendo", "yo", "tú", "él", "ella", "ello",
			"nosotros", "vosotros", "ellos", "ellas", "mi", "tu", "su", "nuestro",
			"vuestro", "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas",
			"aquel", "aquella", "aquellos", "aquellas", "lo", "le", "les", "se", "me",
			"te", "nos", "os", "como", "cuando", "donde", "porque", "si", "no", "sí",
			"más", "menos", "muy", "mucho", "poco", "todo", "nada", "algo", "cada",
			"otro", "mismo", "tan", "tanto", "cual", "quien", "cuyo", "qué", "dónde",
		}...)
	case "fr": // French
		words = append(words, []string{
			"le", "la", "les", "un", "une", "des", "de", "du", "et", "à", "au", "aux",
			"en", "pour", "par", "avec", "sans", "sous", "sur", "dans", "contre",
			"vers", "chez", "entre", "depuis", "pendant", "avant", "après", "devant",
			"derrière", "est", "sont", "être", "avoir", "fait", "faire", "dit", "dire",
			"aller", "voir", "savoir", "pouvoir", "falloir", "vouloir", "je", "tu",
			"il", "elle", "on", "nous", "vous", "ils", "elles", "mon", "ton", "son",
			"ma", "ta", "sa", "mes", "tes", "ses", "notre", "votre", "leur", "nos",
			"vos", "leurs", "ce", "cette", "ces", "celui", "celle", "ceux", "celles",
			"ceci", "cela", "ça", "que", "qui", "quoi", "dont", "où", "si", "ne",
			"pas", "plus", "moins", "très", "bien", "mal", "peu", "beaucoup", "trop",
			"tout", "tous", "toute", "toutes", "quel", "quelle", "quels", "quelles",
			"même", "autre", "aucun", "certain", "plusieurs", "tel", "chaque",
		}...)
	case "de": // German
		words = append(words, []string{
			"der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem",
			"einer", "eines", "und", "oder", "aber", "doch", "sondern", "denn", "weil",
			"wenn", "als", "dass", "ob", "zu", "in", "an", "auf", "aus", "bei", "mit",
			"nach", "von", "vor", "für", "über", "unter", "zwischen", "durch", "gegen",
			"ohne", "um", "bis", "seit", "während", "trotz", "wegen", "ist", "sind",
			"war", "waren", "sein", "haben", "werden", "können", "müssen", "sollen",
			"wollen", "mögen", "dürfen", "ich", "du", "er", "sie", "es", "wir", "ihr",
			"mein", "dein", "sein", "unser", "euer", "dieser", "diese", "dieses",
			"jener", "jene", "jenes", "welcher", "welche", "welches", "man", "sich",
			"nicht", "kein", "keine", "sehr", "schon", "noch", "nur", "auch", "wieder",
			"immer", "nie", "oft", "manchmal", "alle", "alles", "viel", "wenig",
			"mehr", "weniger", "etwas", "nichts", "jemand", "niemand", "wo", "wann",
			"wie", "warum", "was", "wer", "wen", "wem", "wessen",
		}...)
	case "ja": // Japanese (hiragana particles and common words)
		words = append(words, []string{
			"の", "は", "を", "に", "が", "と", "で", "て", "も", "から", "まで",
			"へ", "や", "か", "など", "ね", "よ", "わ", "さ", "これ", "それ",
			"あれ", "この", "その", "あの", "ここ", "そこ", "あそこ", "こう",
			"そう", "ああ", "いる", "ある", "する", "なる", "れる", "られる",
			"せる", "させる", "ない", "ます", "です", "だ", "である", "でも",
			"しかし", "また", "および", "または", "あるいは", "なお", "ただし",
		}...)
	}
	
	return words
}

// NormalizeText performs language-specific text normalization
func (lsp *LanguageSpecificProcessor) NormalizeText(text string) string {
	switch lsp.language {
	case German:
		// Handle German-specific normalizations
		text = strings.ReplaceAll(text, "ß", "ss")
		text = strings.ReplaceAll(text, "ä", "ae")
		text = strings.ReplaceAll(text, "ö", "oe")
		text = strings.ReplaceAll(text, "ü", "ue")
		text = strings.ReplaceAll(text, "Ä", "AE")
		text = strings.ReplaceAll(text, "Ö", "OE")
		text = strings.ReplaceAll(text, "Ü", "UE")
	case French:
		// Handle French-specific normalizations
		text = strings.ReplaceAll(text, "ç", "c")
		text = strings.ReplaceAll(text, "Ç", "C")
		// Remove accents
		replacements := map[string]string{
			"à": "a", "á": "a", "â": "a", "ã": "a", "ä": "a", "å": "a",
			"è": "e", "é": "e", "ê": "e", "ë": "e",
			"ì": "i", "í": "i", "î": "i", "ï": "i",
			"ò": "o", "ó": "o", "ô": "o", "õ": "o", "ö": "o",
			"ù": "u", "ú": "u", "û": "u", "ü": "u",
			"ý": "y", "ÿ": "y",
			"ñ": "n",
		}
		for accented, base := range replacements {
			text = strings.ReplaceAll(text, accented, base)
			text = strings.ReplaceAll(text, strings.ToUpper(accented), strings.ToUpper(base))
		}
	case Spanish:
		// Handle Spanish-specific normalizations
		text = strings.ReplaceAll(text, "ñ", "n")
		text = strings.ReplaceAll(text, "Ñ", "N")
		// Remove accents
		replacements := map[string]string{
			"á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
			"Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
		}
		for accented, base := range replacements {
			text = strings.ReplaceAll(text, accented, base)
		}
	}
	
	return text
}

// GetTokenizationRules returns language-specific tokenization rules
func (lsp *LanguageSpecificProcessor) GetTokenizationRules() []string {
	switch lsp.language {
	case German:
		// German compound word splitting markers
		return []string{"der", "die", "das", "des", "dem", "den"}
	case French:
		// French contractions and liaisons
		return []string{"l'", "d'", "n'", "m'", "t'", "s'", "c'", "qu'"}
	case Spanish:
		// Spanish contractions
		return []string{"al", "del"}
	default:
		return []string{}
	}
}

// MultilingualDocument represents a document that can be processed in multiple languages
type MultilingualDocument struct {
	*Document
	detectedLanguage   Language
	confidence         float64
	languageProcessor  *LanguageSpecificProcessor
}

// NewMultilingualDocument creates a document with automatic language detection
func NewMultilingualDocument(text string, opts ...DocOpt) (*MultilingualDocument, error) {
	detector := NewLanguageDetector()
	detectedLang, confidence := detector.DetectLanguage(text)
	
	// Add language detection to options
	opts = append(opts, WithLanguage(detectedLang))
	
	doc, err := NewDocument(text, opts...)
	if err != nil {
		return nil, err
	}
	
	processor := NewLanguageSpecificProcessor(detectedLang)
	
	return &MultilingualDocument{
		Document:          doc,
		detectedLanguage:  detectedLang,
		confidence:        confidence,
		languageProcessor: processor,
	}, nil
}

// DetectedLanguage returns the detected language and confidence
func (md *MultilingualDocument) DetectedLanguage() (Language, float64) {
	return md.detectedLanguage, md.confidence
}

// GetStopWords returns stop words for the detected language
func (md *MultilingualDocument) GetStopWords() []string {
	return md.languageProcessor.GetStopWords()
}

// NormalizeText normalizes text according to language-specific rules
func (md *MultilingualDocument) NormalizeText(text string) string {
	return md.languageProcessor.NormalizeText(text)
}

// IsMultilingualSupported checks if a language is supported for multilingual processing
func IsMultilingualSupported(lang Language) bool {
	supportedLanguages := []Language{English, Spanish, French, German}
	for _, supported := range supportedLanguages {
		if lang == supported {
			return true
		}
	}
	return false
}

// GetSupportedLanguages returns all supported languages
func GetSupportedLanguages() []Language {
	return []Language{English, Spanish, French, German, Japanese}
}

// FormatLanguageError creates a formatted error for unsupported languages
func FormatLanguageError(lang Language) error {
	return fmt.Errorf("language %s is not supported. Supported languages: %v", 
		string(lang), GetSupportedLanguages())
}