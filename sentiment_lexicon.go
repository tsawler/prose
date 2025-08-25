package prose

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
)

// SentimentLexicon manages sentiment word lists
type SentimentLexicon struct {
	words     map[string]LexiconEntry
	modifiers map[string]float64
	negations map[string]bool
	mutex     sync.RWMutex
}

// LexiconEntry represents a word's sentiment information
type LexiconEntry struct {
	Word       string
	Sentiment  float64  // -1 to 1
	Confidence float64  // 0 to 1
	POS        []string // Applicable POS tags
	Domain     string   // Domain specificity
}

// ExternalLexicon represents the JSON structure for external lexicon files
type ExternalLexicon struct {
	Languages map[string]LanguageLexicon `json:"languages"`
}

// LanguageLexicon contains all word categories for a specific language
type LanguageLexicon struct {
	Words       []WordEntry     `json:"words,omitempty"`
	Modifiers   []ModifierEntry `json:"modifiers,omitempty"`
	Negations   []string        `json:"negations,omitempty"`
	Positive    []WordEntry     `json:"positive,omitempty"`
	Negative    []WordEntry     `json:"negative,omitempty"`
	Intensifiers []string       `json:"intensifiers,omitempty"`
	Diminishers []string        `json:"diminishers,omitempty"`
}

// WordEntry represents a sentiment word in JSON format
type WordEntry struct {
	Word       string   `json:"word"`
	Sentiment  float64  `json:"sentiment"`
	Confidence float64  `json:"confidence"`
	POS        []string `json:"pos,omitempty"`
	Domain     string   `json:"domain,omitempty"`
}

// ModifierEntry represents a modifier word in JSON format
type ModifierEntry struct {
	Word   string  `json:"word"`
	Factor float64 `json:"factor"`
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

// LoadSentimentLexiconWithExternal loads lexicon with optional external file support
func LoadSentimentLexiconWithExternal(lang Language, externalPath string) (*SentimentLexicon, error) {
	lexicon := &SentimentLexicon{
		words:     make(map[string]LexiconEntry),
		modifiers: make(map[string]float64),
		negations: make(map[string]bool),
	}

	// Load base lexicon (hardcoded core words)
	lexicon.loadBaseLexicon(lang)
	lexicon.loadModifiers(lang)
	lexicon.loadNegations(lang)

	// Load external lexicon if provided
	if externalPath != "" {
		if err := lexicon.LoadExternalLexicon(externalPath, []Language{lang}); err != nil {
			return nil, fmt.Errorf("failed to load external lexicon: %w", err)
		}
	}

	return lexicon, nil
}

// LoadExternalLexicon loads and merges external lexicon data
func (sl *SentimentLexicon) LoadExternalLexicon(filepath string, languages []Language) error {
	sl.mutex.Lock()
	defer sl.mutex.Unlock()

	// Read JSON file
	data, err := os.ReadFile(filepath)
	if err != nil {
		return fmt.Errorf("error reading lexicon file: %w", err)
	}

	// Parse JSON
	var external ExternalLexicon
	if err := json.Unmarshal(data, &external); err != nil {
		return fmt.Errorf("error parsing lexicon JSON: %w", err)
	}

	// Process requested languages
	for _, lang := range languages {
		langKey := languageToJSONKey(lang)
		if langData, exists := external.Languages[langKey]; exists {
			sl.mergeLanguageData(langData)
		}
	}

	return nil
}

// languageToJSONKey converts Language constants to JSON keys
func languageToJSONKey(lang Language) string {
	switch lang {
	case English:
		return "english"
	case Spanish:
		return "spanish"
	case French:
		return "french"
	case German:
		return "german"
	case Japanese:
		return "japanese"
	default:
		return strings.ToLower(string(lang))
	}
}

// mergeLanguageData merges external language data with existing lexicon
func (sl *SentimentLexicon) mergeLanguageData(data LanguageLexicon) {
	// Merge sentiment words
	for _, entry := range data.Words {
		sl.words[strings.ToLower(entry.Word)] = LexiconEntry{
			Word:       entry.Word,
			Sentiment:  entry.Sentiment,
			Confidence: entry.Confidence,
			POS:        entry.POS,
			Domain:     entry.Domain,
		}
	}

	// Merge positive words (convert to LexiconEntry format)
	for _, entry := range data.Positive {
		sl.words[strings.ToLower(entry.Word)] = LexiconEntry{
			Word:       entry.Word,
			Sentiment:  entry.Sentiment,
			Confidence: entry.Confidence,
			POS:        entry.POS,
			Domain:     entry.Domain,
		}
	}

	// Merge negative words (convert to LexiconEntry format)
	for _, entry := range data.Negative {
		sl.words[strings.ToLower(entry.Word)] = LexiconEntry{
			Word:       entry.Word,
			Sentiment:  entry.Sentiment,
			Confidence: entry.Confidence,
			POS:        entry.POS,
			Domain:     entry.Domain,
		}
	}

	// Merge modifiers
	for _, modifier := range data.Modifiers {
		sl.modifiers[strings.ToLower(modifier.Word)] = modifier.Factor
	}

	// Merge intensifiers (default factor)
	for _, intensifier := range data.Intensifiers {
		sl.modifiers[strings.ToLower(intensifier)] = 1.5 // Default intensifier factor
	}

	// Merge diminishers (default factor)
	for _, diminisher := range data.Diminishers {
		sl.modifiers[strings.ToLower(diminisher)] = 0.5 // Default diminisher factor
	}

	// Merge negations
	for _, negation := range data.Negations {
		sl.negations[strings.ToLower(negation)] = true
	}
}

// loadBaseLexicon loads sentiment words
func (sl *SentimentLexicon) loadBaseLexicon(lang Language) {
	// Language-specific lexicons
	switch lang {
	case English:
		sl.loadEnglishLexicon()
	case Spanish:
		sl.loadSpanishLexicon()
	case French:
		sl.loadFrenchLexicon()
	case German:
		sl.loadGermanLexicon()
	case Japanese:
		sl.loadJapaneseLexicon()
	default:
		// Default to English
		sl.loadEnglishLexicon()
	}
}

// loadEnglishLexicon loads English sentiment words
func (sl *SentimentLexicon) loadEnglishLexicon() {
	sl.words = map[string]LexiconEntry{
		// Strong positive words
		"excellent":   {Word: "excellent", Sentiment: 0.9, Confidence: 0.95},
		"amazing":     {Word: "amazing", Sentiment: 0.85, Confidence: 0.95},
		"wonderful":   {Word: "wonderful", Sentiment: 0.85, Confidence: 0.95},
		"fantastic":   {Word: "fantastic", Sentiment: 0.85, Confidence: 0.95},
		"outstanding": {Word: "outstanding", Sentiment: 0.9, Confidence: 0.95},
		"perfect":     {Word: "perfect", Sentiment: 0.95, Confidence: 0.95},
		"brilliant":   {Word: "brilliant", Sentiment: 0.85, Confidence: 0.95},
		"superb":      {Word: "superb", Sentiment: 0.85, Confidence: 0.95},
		"magnificent": {Word: "magnificent", Sentiment: 0.9, Confidence: 0.95},
		
		// Moderate positive words
		"good":       {Word: "good", Sentiment: 0.6, Confidence: 0.9},
		"great":      {Word: "great", Sentiment: 0.75, Confidence: 0.9},
		"nice":       {Word: "nice", Sentiment: 0.5, Confidence: 0.85},
		"love":       {Word: "love", Sentiment: 0.8, Confidence: 0.9},
		"happy":      {Word: "happy", Sentiment: 0.7, Confidence: 0.9},
		"beautiful":  {Word: "beautiful", Sentiment: 0.75, Confidence: 0.9},
		"enjoy":      {Word: "enjoy", Sentiment: 0.65, Confidence: 0.9},
		"like":       {Word: "like", Sentiment: 0.5, Confidence: 0.85},
		"pleasant":   {Word: "pleasant", Sentiment: 0.6, Confidence: 0.9},
		"positive":   {Word: "positive", Sentiment: 0.6, Confidence: 0.9},
		"best":       {Word: "best", Sentiment: 0.85, Confidence: 0.95},
		"better":     {Word: "better", Sentiment: 0.5, Confidence: 0.85},
		"fun":        {Word: "fun", Sentiment: 0.65, Confidence: 0.9},
		"interesting":{Word: "interesting", Sentiment: 0.5, Confidence: 0.85},
		"awesome":    {Word: "awesome", Sentiment: 0.8, Confidence: 0.9},
		
		// Mild positive words
		"okay":       {Word: "okay", Sentiment: 0.2, Confidence: 0.7},
		"fine":       {Word: "fine", Sentiment: 0.3, Confidence: 0.75},
		"decent":     {Word: "decent", Sentiment: 0.4, Confidence: 0.8},
		"satisfactory": {Word: "satisfactory", Sentiment: 0.4, Confidence: 0.85},
		
		// Strong negative words
		"terrible":    {Word: "terrible", Sentiment: -0.9, Confidence: 0.95},
		"awful":       {Word: "awful", Sentiment: -0.85, Confidence: 0.95},
		"horrible":    {Word: "horrible", Sentiment: -0.85, Confidence: 0.95},
		"disgusting":  {Word: "disgusting", Sentiment: -0.9, Confidence: 0.95},
		"appalling":   {Word: "appalling", Sentiment: -0.9, Confidence: 0.95},
		"dreadful":    {Word: "dreadful", Sentiment: -0.85, Confidence: 0.95},
		"atrocious":   {Word: "atrocious", Sentiment: -0.9, Confidence: 0.95},
		"abysmal":     {Word: "abysmal", Sentiment: -0.95, Confidence: 0.95},
		
		// Moderate negative words
		"bad":           {Word: "bad", Sentiment: -0.6, Confidence: 0.9},
		"hate":          {Word: "hate", Sentiment: -0.8, Confidence: 0.9},
		"sad":           {Word: "sad", Sentiment: -0.7, Confidence: 0.9},
		"ugly":          {Word: "ugly", Sentiment: -0.75, Confidence: 0.9},
		"disappointing": {Word: "disappointing", Sentiment: -0.7, Confidence: 0.9},
		"poor":          {Word: "poor", Sentiment: -0.65, Confidence: 0.9},
		"wrong":         {Word: "wrong", Sentiment: -0.6, Confidence: 0.85},
		"worst":         {Word: "worst", Sentiment: -0.85, Confidence: 0.95},
		"worse":         {Word: "worse", Sentiment: -0.5, Confidence: 0.85},
		"dislike":       {Word: "dislike", Sentiment: -0.5, Confidence: 0.85},
		"negative":      {Word: "negative", Sentiment: -0.6, Confidence: 0.9},
		"annoying":      {Word: "annoying", Sentiment: -0.65, Confidence: 0.9},
		"boring":        {Word: "boring", Sentiment: -0.6, Confidence: 0.85},
		"fail":          {Word: "fail", Sentiment: -0.7, Confidence: 0.9},
		"failure":       {Word: "failure", Sentiment: -0.75, Confidence: 0.9},
		
		// Context-dependent words
		"cheap":      {Word: "cheap", Sentiment: -0.3, Confidence: 0.6},
		"simple":     {Word: "simple", Sentiment: 0.1, Confidence: 0.5},
		"long":       {Word: "long", Sentiment: 0.0, Confidence: 0.3},
		"short":      {Word: "short", Sentiment: 0.0, Confidence: 0.3},
		"fast":       {Word: "fast", Sentiment: 0.3, Confidence: 0.6},
		"slow":       {Word: "slow", Sentiment: -0.3, Confidence: 0.6},
		"hard":       {Word: "hard", Sentiment: -0.2, Confidence: 0.5},
		"easy":       {Word: "easy", Sentiment: 0.3, Confidence: 0.6},
		"complex":    {Word: "complex", Sentiment: -0.1, Confidence: 0.4},
		"new":        {Word: "new", Sentiment: 0.2, Confidence: 0.5},
		"old":        {Word: "old", Sentiment: -0.2, Confidence: 0.5},
	}
}

// loadSpanishLexicon loads Spanish sentiment words
func (sl *SentimentLexicon) loadSpanishLexicon() {
	sl.words = map[string]LexiconEntry{
		// Positive words
		"excelente":   {Word: "excelente", Sentiment: 0.9, Confidence: 0.95},
		"maravilloso": {Word: "maravilloso", Sentiment: 0.85, Confidence: 0.95},
		"fantástico":  {Word: "fantástico", Sentiment: 0.85, Confidence: 0.95},
		"bueno":       {Word: "bueno", Sentiment: 0.6, Confidence: 0.9},
		"genial":      {Word: "genial", Sentiment: 0.75, Confidence: 0.9},
		"amor":        {Word: "amor", Sentiment: 0.8, Confidence: 0.9},
		"feliz":       {Word: "feliz", Sentiment: 0.7, Confidence: 0.9},
		"hermoso":     {Word: "hermoso", Sentiment: 0.75, Confidence: 0.9},
		"mejor":       {Word: "mejor", Sentiment: 0.5, Confidence: 0.85},
		
		// Negative words
		"terrible":      {Word: "terrible", Sentiment: -0.9, Confidence: 0.95},
		"horrible":      {Word: "horrible", Sentiment: -0.85, Confidence: 0.95},
		"malo":          {Word: "malo", Sentiment: -0.6, Confidence: 0.9},
		"odio":          {Word: "odio", Sentiment: -0.8, Confidence: 0.9},
		"triste":        {Word: "triste", Sentiment: -0.7, Confidence: 0.9},
		"feo":           {Word: "feo", Sentiment: -0.75, Confidence: 0.9},
		"decepcionante": {Word: "decepcionante", Sentiment: -0.7, Confidence: 0.9},
		"peor":          {Word: "peor", Sentiment: -0.5, Confidence: 0.85},
	}
}

// loadFrenchLexicon loads French sentiment words
func (sl *SentimentLexicon) loadFrenchLexicon() {
	sl.words = map[string]LexiconEntry{
		// Positive words
		"excellent":   {Word: "excellent", Sentiment: 0.9, Confidence: 0.95},
		"merveilleux": {Word: "merveilleux", Sentiment: 0.85, Confidence: 0.95},
		"fantastique": {Word: "fantastique", Sentiment: 0.85, Confidence: 0.95},
		"bon":         {Word: "bon", Sentiment: 0.6, Confidence: 0.9},
		"génial":      {Word: "génial", Sentiment: 0.75, Confidence: 0.9},
		"amour":       {Word: "amour", Sentiment: 0.8, Confidence: 0.9},
		"heureux":     {Word: "heureux", Sentiment: 0.7, Confidence: 0.9},
		"beau":        {Word: "beau", Sentiment: 0.75, Confidence: 0.9},
		"meilleur":    {Word: "meilleur", Sentiment: 0.5, Confidence: 0.85},
		
		// Negative words
		"terrible":   {Word: "terrible", Sentiment: -0.9, Confidence: 0.95},
		"horrible":   {Word: "horrible", Sentiment: -0.85, Confidence: 0.95},
		"mauvais":    {Word: "mauvais", Sentiment: -0.6, Confidence: 0.9},
		"déteste":    {Word: "déteste", Sentiment: -0.8, Confidence: 0.9},
		"triste":     {Word: "triste", Sentiment: -0.7, Confidence: 0.9},
		"laid":       {Word: "laid", Sentiment: -0.75, Confidence: 0.9},
		"décevant":   {Word: "décevant", Sentiment: -0.7, Confidence: 0.9},
		"pire":       {Word: "pire", Sentiment: -0.5, Confidence: 0.85},
	}
}

// loadGermanLexicon loads German sentiment words
func (sl *SentimentLexicon) loadGermanLexicon() {
	sl.words = map[string]LexiconEntry{
		// Positive words
		"ausgezeichnet": {Word: "ausgezeichnet", Sentiment: 0.9, Confidence: 0.95},
		"wunderbar":     {Word: "wunderbar", Sentiment: 0.85, Confidence: 0.95},
		"fantastisch":   {Word: "fantastisch", Sentiment: 0.85, Confidence: 0.95},
		"gut":           {Word: "gut", Sentiment: 0.6, Confidence: 0.9},
		"großartig":     {Word: "großartig", Sentiment: 0.75, Confidence: 0.9},
		"liebe":         {Word: "liebe", Sentiment: 0.8, Confidence: 0.9},
		"glücklich":     {Word: "glücklich", Sentiment: 0.7, Confidence: 0.9},
		"schön":         {Word: "schön", Sentiment: 0.75, Confidence: 0.9},
		"besser":        {Word: "besser", Sentiment: 0.5, Confidence: 0.85},
		
		// Negative words
		"schrecklich":   {Word: "schrecklich", Sentiment: -0.9, Confidence: 0.95},
		"furchtbar":     {Word: "furchtbar", Sentiment: -0.85, Confidence: 0.95},
		"schlecht":      {Word: "schlecht", Sentiment: -0.6, Confidence: 0.9},
		"hasse":         {Word: "hasse", Sentiment: -0.8, Confidence: 0.9},
		"traurig":       {Word: "traurig", Sentiment: -0.7, Confidence: 0.9},
		"hässlich":      {Word: "hässlich", Sentiment: -0.75, Confidence: 0.9},
		"enttäuschend":  {Word: "enttäuschend", Sentiment: -0.7, Confidence: 0.9},
		"schlechter":    {Word: "schlechter", Sentiment: -0.5, Confidence: 0.85},
	}
}

// loadJapaneseLexicon loads Japanese sentiment words
func (sl *SentimentLexicon) loadJapaneseLexicon() {
	sl.words = map[string]LexiconEntry{
		// Positive words (Hiragana/Katakana/Kanji)
		"良い":     {Word: "良い", Sentiment: 0.6, Confidence: 0.9},      // yoi (good)
		"いい":     {Word: "いい", Sentiment: 0.6, Confidence: 0.9},      // ii (good)
		"素晴らしい": {Word: "素晴らしい", Sentiment: 0.85, Confidence: 0.95}, // subarashii (wonderful)
		"すごい":   {Word: "すごい", Sentiment: 0.75, Confidence: 0.9},    // sugoi (amazing)
		"大好き":   {Word: "大好き", Sentiment: 0.8, Confidence: 0.9},     // daisuki (love)
		"嬉しい":   {Word: "嬉しい", Sentiment: 0.7, Confidence: 0.9},     // ureshii (happy)
		"美しい":   {Word: "美しい", Sentiment: 0.75, Confidence: 0.9},    // utsukushii (beautiful)
		"完璧":    {Word: "完璧", Sentiment: 0.9, Confidence: 0.95},     // kanpeki (perfect)
		"最高":    {Word: "最高", Sentiment: 0.85, Confidence: 0.95},    // saikou (best/greatest)
		"楽しい":   {Word: "楽しい", Sentiment: 0.7, Confidence: 0.9},     // tanoshii (fun/enjoyable)
		
		// Negative words
		"悪い":    {Word: "悪い", Sentiment: -0.6, Confidence: 0.9},     // warui (bad)
		"ひどい":   {Word: "ひどい", Sentiment: -0.8, Confidence: 0.9},    // hidoi (terrible)
		"嫌い":    {Word: "嫌い", Sentiment: -0.7, Confidence: 0.9},     // kirai (dislike/hate)
		"悲しい":   {Word: "悲しい", Sentiment: -0.7, Confidence: 0.9},    // kanashii (sad)
		"つまらない": {Word: "つまらない", Sentiment: -0.6, Confidence: 0.85}, // tsumaranai (boring)
		"最悪":    {Word: "最悪", Sentiment: -0.85, Confidence: 0.95},   // saiaku (worst)
		"残念":    {Word: "残念", Sentiment: -0.6, Confidence: 0.85},    // zannen (disappointing)
		"怖い":    {Word: "怖い", Sentiment: -0.65, Confidence: 0.9},    // kowai (scary/frightening)
	}
}

// loadModifiers loads intensifiers and diminishers
func (sl *SentimentLexicon) loadModifiers(lang Language) {
	switch lang {
	case English:
		sl.loadEnglishModifiers()
	case Spanish:
		sl.loadSpanishModifiers()
	case French:
		sl.loadFrenchModifiers()
	case German:
		sl.loadGermanModifiers()
	case Japanese:
		sl.loadJapaneseModifiers()
	default:
		sl.loadEnglishModifiers()
	}
}

// loadEnglishModifiers loads English modifiers
func (sl *SentimentLexicon) loadEnglishModifiers() {
	sl.modifiers = map[string]float64{
		// Intensifiers (increase by factor)
		"very":       0.3,
		"extremely":  0.5,
		"absolutely": 0.5,
		"totally":    0.4,
		"really":     0.3,
		"so":         0.3,
		"quite":      0.2,
		"incredibly": 0.5,
		"remarkably": 0.4,
		"particularly": 0.3,
		"especially": 0.3,
		"super":      0.4,
		"utterly":    0.5,
		"completely": 0.4,
		"thoroughly": 0.4,
		
		// Diminishers (decrease by factor)
		"slightly":   -0.3,
		"somewhat":   -0.3,
		"rather":     -0.2,
		"fairly":     -0.1,
		"marginally": -0.4,
		"barely":     -0.5,
		"hardly":     -0.5,
		"scarcely":   -0.5,
		"a bit":      -0.2,
		"a little":   -0.2,
		"kind of":    -0.3,
		"sort of":    -0.3,
	}
}

// loadSpanishModifiers loads Spanish modifiers
func (sl *SentimentLexicon) loadSpanishModifiers() {
	sl.modifiers = map[string]float64{
		// Intensifiers
		"muy":           0.3,
		"extremadamente": 0.5,
		"absolutamente": 0.5,
		"totalmente":    0.4,
		"realmente":     0.3,
		"bastante":      0.2,
		"súper":         0.4,
		
		// Diminishers
		"ligeramente": -0.3,
		"algo":        -0.3,
		"poco":        -0.3,
		"apenas":      -0.5,
	}
}

// loadFrenchModifiers loads French modifiers
func (sl *SentimentLexicon) loadFrenchModifiers() {
	sl.modifiers = map[string]float64{
		// Intensifiers
		"très":         0.3,
		"extrêmement":  0.5,
		"absolument":   0.5,
		"totalement":   0.4,
		"vraiment":     0.3,
		"assez":        0.2,
		"super":        0.4,
		
		// Diminishers
		"légèrement": -0.3,
		"quelque peu": -0.3,
		"peu":        -0.3,
		"à peine":    -0.5,
	}
}

// loadGermanModifiers loads German modifiers
func (sl *SentimentLexicon) loadGermanModifiers() {
	sl.modifiers = map[string]float64{
		// Intensifiers
		"sehr":      0.3,
		"extrem":    0.5,
		"absolut":   0.5,
		"total":     0.4,
		"wirklich":  0.3,
		"ziemlich":  0.2,
		"super":     0.4,
		
		// Diminishers
		"leicht":    -0.3,
		"etwas":     -0.3,
		"wenig":     -0.3,
		"kaum":      -0.5,
	}
}

// loadJapaneseModifiers loads Japanese modifiers
func (sl *SentimentLexicon) loadJapaneseModifiers() {
	sl.modifiers = map[string]float64{
		// Intensifiers
		"とても":   0.3,  // totemo (very)
		"すごく":   0.4,  // sugoku (very/extremely)
		"非常に":   0.5,  // hijou ni (extremely)
		"本当に":   0.3,  // hontou ni (really)
		"かなり":   0.2,  // kanari (quite/fairly)
		"めちゃくちゃ": 0.5,  // mechakucha (extremely/very)
		"超":     0.4,  // chou (super)
		"完全に":   0.5,  // kanzen ni (completely)
		
		// Diminishers
		"少し":    -0.3, // sukoshi (a little)
		"ちょっと":  -0.3, // chotto (a little/slightly)
		"やや":    -0.2, // yaya (somewhat)
		"わずかに":  -0.4, // wazuka ni (slightly)
		"あまり":   -0.4, // amari (not very) - when used with negative
	}
}

// loadNegations loads negation words
func (sl *SentimentLexicon) loadNegations(lang Language) {
	switch lang {
	case English:
		sl.loadEnglishNegations()
	case Spanish:
		sl.loadSpanishNegations()
	case French:
		sl.loadFrenchNegations()
	case German:
		sl.loadGermanNegations()
	case Japanese:
		sl.loadJapaneseNegations()
	default:
		sl.loadEnglishNegations()
	}
}

// loadEnglishNegations loads English negation words
func (sl *SentimentLexicon) loadEnglishNegations() {
	sl.negations = map[string]bool{
		"not":      true,
		"no":       true,
		"never":    true,
		"neither":  true,
		"nor":      true,
		"cannot":   true,
		"can't":    true,
		"won't":    true,
		"don't":    true,
		"doesn't":  true,
		"didn't":   true,
		"isn't":    true,
		"aren't":   true,
		"wasn't":   true,
		"weren't":  true,
		"hasn't":   true,
		"haven't":  true,
		"hadn't":   true,
		"wouldn't": true,
		"shouldn't": true,
		"couldn't": true,
		"without":  true,
		"nobody":   true,
		"nothing":  true,
		"nowhere":  true,
		"none":     true,
	}
}

// loadSpanishNegations loads Spanish negation words
func (sl *SentimentLexicon) loadSpanishNegations() {
	sl.negations = map[string]bool{
		"no":     true,
		"nunca":  true,
		"jamás":  true,
		"ni":     true,
		"sin":    true,
		"nada":   true,
		"nadie":  true,
		"ningún": true,
		"ninguna": true,
		"tampoco": true,
	}
}

// loadFrenchNegations loads French negation words
func (sl *SentimentLexicon) loadFrenchNegations() {
	sl.negations = map[string]bool{
		"ne":      true,
		"pas":     true,
		"non":     true,
		"jamais":  true,
		"rien":    true,
		"personne": true,
		"aucun":   true,
		"aucune":  true,
		"ni":      true,
		"sans":    true,
	}
}

// loadGermanNegations loads German negation words
func (sl *SentimentLexicon) loadGermanNegations() {
	sl.negations = map[string]bool{
		"nicht":   true,
		"nein":    true,
		"kein":    true,
		"keine":   true,
		"niemals": true,
		"nie":     true,
		"nichts":  true,
		"niemand": true,
		"nirgends": true,
		"ohne":    true,
	}
}

// loadJapaneseNegations loads Japanese negation words
func (sl *SentimentLexicon) loadJapaneseNegations() {
	sl.negations = map[string]bool{
		"ない":   true, // nai (not/there is not)
		"いない":  true, // inai (not present/not there) 
		"ではない": true, // dewa nai (is not)
		"じゃない": true, // ja nai (isn't) - casual
		"しない":  true, // shinai (don't do)
		"できない": true, // dekinai (cannot)
		"わからない": true, // wakaranai (don't know/don't understand)
		"だめ":   true, // dame (no good/not allowed)
		"いけない": true, // ikenai (must not)
		"なし":   true, // nashi (without/none)
		"絶対":   true, // zettai (absolutely) - when used with negative
		"決して":  true, // kesshite (never)
		"全然":   true, // zenzen (not at all) - when used with negative
	}
}

// GetSentiment returns sentiment score for a word
func (sl *SentimentLexicon) GetSentiment(word string) float64 {
	sl.mutex.RLock()
	defer sl.mutex.RUnlock()

	// Try exact match first
	if entry, exists := sl.words[word]; exists {
		return entry.Sentiment
	}
	
	// Try lowercase
	if entry, exists := sl.words[strings.ToLower(word)]; exists {
		return entry.Sentiment
	}
	
	return 0.0
}

// GetConfidence returns confidence for a word's sentiment
func (sl *SentimentLexicon) GetConfidence(word string) float64 {
	sl.mutex.RLock()
	defer sl.mutex.RUnlock()

	// Try exact match first
	if entry, exists := sl.words[word]; exists {
		return entry.Confidence
	}
	
	// Try lowercase
	if entry, exists := sl.words[strings.ToLower(word)]; exists {
		return entry.Confidence
	}
	
	return 0.0
}

// IsNegation checks if word is a negation
func (sl *SentimentLexicon) IsNegation(word string) bool {
	sl.mutex.RLock()
	defer sl.mutex.RUnlock()

	// Check exact match and lowercase
	return sl.negations[word] || sl.negations[strings.ToLower(word)]
}

// GetModifierStrength returns modifier strength
func (sl *SentimentLexicon) GetModifierStrength(word string) float64 {
	sl.mutex.RLock()
	defer sl.mutex.RUnlock()

	// Try exact match first
	if strength, exists := sl.modifiers[word]; exists {
		return strength
	}
	
	// Try lowercase
	if strength, exists := sl.modifiers[strings.ToLower(word)]; exists {
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

// AddCustomModifier adds a custom modifier
func (sl *SentimentLexicon) AddCustomModifier(word string, strength float64) {
	sl.mutex.Lock()
	defer sl.mutex.Unlock()

	sl.modifiers[word] = strength
}

// AddCustomNegation adds a custom negation word
func (sl *SentimentLexicon) AddCustomNegation(word string) {
	sl.mutex.Lock()
	defer sl.mutex.Unlock()

	sl.negations[word] = true
}

// GetLexiconSize returns the number of words in the lexicon
func (sl *SentimentLexicon) GetLexiconSize() int {
	sl.mutex.RLock()
	defer sl.mutex.RUnlock()

	return len(sl.words)
}

// HasWord checks if a word exists in the lexicon
func (sl *SentimentLexicon) HasWord(word string) bool {
	sl.mutex.RLock()
	defer sl.mutex.RUnlock()

	_, exists := sl.words[word]
	if !exists {
		_, exists = sl.words[strings.ToLower(word)]
	}
	return exists
}