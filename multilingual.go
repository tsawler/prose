package prose

import (
	"fmt"
	"regexp"
	"strings"
	"unicode"
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
	switch lsp.language {
	case English:
		return []string{
			"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in", "is", "it",
			"its", "of", "on", "that", "the", "to", "was", "will", "with", "the", "this", "but", "they",
			"have", "had", "what", "said", "each", "which", "she", "do", "how", "their", "if", "up", "out",
			"many", "then", "them", "these", "so", "some", "her", "would", "make", "like", "into", "him",
			"time", "two", "more", "go", "no", "way", "could", "my", "than", "first", "been", "call", "who",
			"its", "now", "find", "long", "down", "day", "did", "get", "come", "made", "may", "part",
		}
	case Spanish:
		return []string{
			"a", "ante", "bajo", "cabe", "con", "contra", "de", "desde", "durante", "en", "entre", "hacia",
			"hasta", "mediante", "para", "por", "según", "sin", "so", "sobre", "tras", "el", "la", "los",
			"las", "un", "una", "unos", "unas", "y", "o", "pero", "si", "no", "que", "como", "cuando",
			"donde", "quien", "cual", "cuyo", "este", "esta", "estos", "estas", "ese", "esa", "esos",
			"esas", "aquel", "aquella", "aquellos", "aquellas", "mi", "tu", "su", "nuestro", "vuestro",
			"mio", "tuyo", "suyo", "me", "te", "se", "nos", "os", "le", "les", "lo", "la", "los", "las",
		}
	case French:
		return []string{
			"le", "de", "et", "à", "un", "il", "être", "et", "en", "avoir", "que", "pour", "dans", "ce",
			"son", "une", "sur", "avec", "ne", "se", "pas", "tout", "plus", "par", "grand", "en", "me",
			"même", "elle", "vous", "ou", "du", "au", "très", "nous", "mon", "comme", "mais", "pouvoir",
			"quel", "temps", "petit", "celui", "type", "cadet", "si", "aujourd", "gros", "si", "contre",
			"pendant", "chez", "entre", "sous", "jusqu", "sans", "vers", "chez", "malgré", "concernant",
		}
	case German:
		return []string{
			"der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich", "des", "auf", "für",
			"ist", "im", "dem", "nicht", "ein", "eine", "als", "auch", "es", "an", "werden", "aus", "er",
			"hat", "dass", "sie", "nach", "wird", "bei", "einer", "um", "am", "sind", "noch", "wie",
			"einem", "über", "einen", "so", "zum", "war", "haben", "nur", "oder", "aber", "vor", "zur",
			"bis", "mehr", "durch", "man", "sein", "wurde", "sei", "in", "gegen", "vom", "können", "schon",
		}
	default:
		return []string{}
	}
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