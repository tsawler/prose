package prose

import (
	"regexp"
	"strings"
	"unicode"
	"unicode/utf8"
)

type TokenTester func(string) bool

type Tokenizer interface {
	Tokenize(string) []*Token
	TokenizeWithOffsets(string) []*Token
}

// iterTokenizer splits a sentence into words.
type iterTokenizer struct {
	specialRE      *regexp.Regexp
	sanitizer      *strings.Replacer
	contractions   []string
	splitCases     []string
	suffixes       []string
	prefixes       []string
	emoticons      map[string]int
	isUnsplittable TokenTester
	tokenPool      *TokenPool
}

type TokenizerOptFunc func(*iterTokenizer)

// UsingIsUnsplittableFN gives a function that tests whether a token is splittable or not.
func UsingIsUnsplittable(x TokenTester) TokenizerOptFunc {
	return func(tokenizer *iterTokenizer) {
		tokenizer.isUnsplittable = x
	}
}

// Use the provided special regex for unsplittable tokens.
func UsingSpecialRE(x *regexp.Regexp) TokenizerOptFunc {
	return func(tokenizer *iterTokenizer) {
		tokenizer.specialRE = x
	}
}

// Use the provided sanitizer.
func UsingSanitizer(x *strings.Replacer) TokenizerOptFunc {
	return func(tokenizer *iterTokenizer) {
		tokenizer.sanitizer = x
	}
}

// Use the provided suffixes.
func UsingSuffixes(x []string) TokenizerOptFunc {
	return func(tokenizer *iterTokenizer) {
		tokenizer.suffixes = x
	}
}

// Use the provided prefixes.
func UsingPrefixes(x []string) TokenizerOptFunc {
	return func(tokenizer *iterTokenizer) {
		tokenizer.prefixes = x
	}
}

// Use the provided map of emoticons.
func UsingEmoticons(x map[string]int) TokenizerOptFunc {
	return func(tokenizer *iterTokenizer) {
		tokenizer.emoticons = x
	}
}

// Use the provided contractions.
func UsingContractions(x []string) TokenizerOptFunc {
	return func(tokenizer *iterTokenizer) {
		tokenizer.contractions = x
	}
}

// Use the provided splitCases.
func UsingSplitCases(x []string) TokenizerOptFunc {
	return func(tokenizer *iterTokenizer) {
		tokenizer.splitCases = x
	}
}

// UsingTokenPool sets the token pool for memory optimization.
func UsingTokenPool(pool *TokenPool) TokenizerOptFunc {
	return func(tokenizer *iterTokenizer) {
		tokenizer.tokenPool = pool
	}
}

// Constructor for default iterTokenizer
func NewIterTokenizer(opts ...TokenizerOptFunc) *iterTokenizer {
	tok := new(iterTokenizer)

	// Set default parameters
	tok.contractions = contractions
	tok.emoticons = emoticons
	tok.isUnsplittable = func(_ string) bool { return false }
	tok.prefixes = prefixes
	tok.sanitizer = sanitizer
	tok.specialRE = internalRE
	tok.suffixes = suffixes
	tok.tokenPool = NewTokenPool()

	// Apply options if provided
	for _, applyOpt := range opts {
		applyOpt(tok)
	}

	tok.splitCases = append(tok.splitCases, tok.contractions...)

	return tok
}

func addToken(s string, toks []*Token) []*Token {
	if strings.TrimSpace(s) != "" {
		toks = append(toks, &Token{Text: s})
	}
	return toks
}

func (t *iterTokenizer) addTokenWithOffset(s string, start int, toks []*Token) []*Token {
	if strings.TrimSpace(s) != "" {
		token := t.tokenPool.Get()
		token.Text = s
		token.Start = start
		token.End = start + len(s)
		toks = append(toks, token)
	}
	return toks
}

func (t *iterTokenizer) isSpecial(token string) bool {
	_, found := t.emoticons[token]
	return found || t.specialRE.MatchString(token) || t.isUnsplittable(token)
}

func (t *iterTokenizer) doSplit(token string) []*Token {
	tokens := []*Token{}
	suffs := []*Token{}

	last := 0
	for token != "" && utf8.RuneCountInString(token) != last {
		if t.isSpecial(token) {
			// We've found a special case (e.g., an emoticon) -- so, we add it as a token without
			// any further processing.
			tokens = addToken(token, tokens)
			break
		}
		last = utf8.RuneCountInString(token)
		lower := strings.ToLower(token)
		if hasAnyPrefix(token, t.prefixes) {
			// Remove prefixes -- e.g., $100 -> [$, 100].
			tokens = addToken(string(token[0]), tokens)
			token = token[1:]
		} else if idx := hasAnyIndex(lower, t.splitCases); idx > -1 {
			// Handle "they'll", "I'll", "Don't", "won't", amount($).
			//
			// they'll -> [they, 'll].
			// don't -> [do, n't].
			// amount($) -> [amount, (, $, )].
			tokens = addToken(token[:idx], tokens)
			token = token[idx:]
		} else if hasAnySuffix(token, t.suffixes) {
			// Remove suffixes -- e.g., Well) -> [Well, )].
			suffs = append([]*Token{
				{Text: string(token[len(token)-1])}},
				suffs...)
			token = token[:len(token)-1]
		} else {
			tokens = addToken(token, tokens)
		}
	}

	return append(tokens, suffs...)
}

func (t *iterTokenizer) doSplitWithOffsets(token string, baseOffset int) []*Token {
	tokens := []*Token{}
	suffs := []*Token{}
	currentOffset := baseOffset

	last := 0
	for token != "" && utf8.RuneCountInString(token) != last {
		if t.isSpecial(token) {
			// We've found a special case (e.g., an emoticon) -- so, we add it as a token without
			// any further processing.
			tokens = t.addTokenWithOffset(token, currentOffset, tokens)
			break
		}
		last = utf8.RuneCountInString(token)
		lower := strings.ToLower(token)
		if hasAnyPrefix(token, t.prefixes) {
			// Remove prefixes -- e.g., $100 -> [$, 100].
			tokens = t.addTokenWithOffset(string(token[0]), currentOffset, tokens)
			token = token[1:]
			currentOffset++
		} else if idx := hasAnyIndex(lower, t.splitCases); idx > -1 {
			// Handle "they'll", "I'll", "Don't", "won't", amount($).
			//
			// they'll -> [they, 'll].
			// don't -> [do, n't].
			// amount($) -> [amount, (, $, )].
			tokens = t.addTokenWithOffset(token[:idx], currentOffset, tokens)
			currentOffset += idx
			token = token[idx:]
		} else if hasAnySuffix(token, t.suffixes) {
			// Remove suffixes -- e.g., Well) -> [Well, )].
			suffixStart := currentOffset + len(token) - 1
			suffixToken := t.tokenPool.Get()
			suffixToken.Text = string(token[len(token)-1])
			suffixToken.Start = suffixStart
			suffixToken.End = suffixStart + 1
			suffs = append([]*Token{suffixToken}, suffs...)
			token = token[:len(token)-1]
		} else {
			tokens = t.addTokenWithOffset(token, currentOffset, tokens)
			break
		}
	}

	return append(tokens, suffs...)
}

// tokenize splits a sentence into a slice of words.
func (t *iterTokenizer) Tokenize(text string) []*Token {
	return t.TokenizeWithOffsets(text)
}

// TokenizeWithOffsets splits a sentence into a slice of words with position tracking.
func (t *iterTokenizer) TokenizeWithOffsets(text string) []*Token {
	var tokens []*Token

	clean, white := t.sanitizer.Replace(text), false
	length := len(clean)

	start, index := 0, 0
	cache := map[string][]*Token{}
	originalIndex := 0
	
	for index <= length {
		uc, size := utf8.DecodeRuneInString(clean[index:])
		if size == 0 {
			break
		} else if index == 0 {
			white = unicode.IsSpace(uc)
		}
		if unicode.IsSpace(uc) != white {
			if start < index {
				span := clean[start:index]
				spanStart := originalIndex
				
				if toks, found := cache[span]; found {
					// Clone tokens and update offsets
					for _, tok := range toks {
						newTok := t.tokenPool.Get()
						newTok.Text = tok.Text
						newTok.Start = spanStart
						newTok.End = spanStart + len(tok.Text)
						tokens = append(tokens, newTok)
						spanStart += len(tok.Text)
					}
				} else {
					toks := t.doSplitWithOffsets(span, spanStart)
					cache[span] = toks
					tokens = append(tokens, toks...)
				}
				originalIndex += index - start
			}
			if uc == ' ' {
				start = index + 1
				originalIndex = index + 1
			} else {
				start = index
				originalIndex = index
			}
			white = !white
		}
		index += size
	}

	if start < index {
		tokens = append(tokens, t.doSplitWithOffsets(clean[start:index], originalIndex)...)
	}

	return tokens
}

var internalRE = regexp.MustCompile(`^(?:[A-Za-z]\.){2,}$|^[A-Z][a-z]{1,2}\.$`)
var sanitizer = strings.NewReplacer(
	"\u201c", `"`,
	"\u201d", `"`,
	"\u2018", "'",
	"\u2019", "'",
	"&rsquo;", "'")
var contractions = []string{"'ll", "'s", "'re", "'m", "n't"}
var suffixes = []string{",", ")", `"`, "]", "!", ";", ".", "?", ":", "'"}
var prefixes = []string{"$", "(", `"`, "["}
var emoticons = map[string]int{
	"(-8":         1,
	"(-;":         1,
	"(-_-)":       1,
	"(._.)":       1,
	"(:":          1,
	"(=":          1,
	"(o:":         1,
	"(¬_¬)":       1,
	"(ಠ_ಠ)":       1,
	"(╯°□°）╯︵┻━┻": 1,
	"-__-":        1,
	"8-)":         1,
	"8-D":         1,
	"8D":          1,
	":(":          1,
	":((":         1,
	":(((":        1,
	":()":         1,
	":)))":        1,
	":-)":         1,
	":-))":        1,
	":-)))":       1,
	":-*":         1,
	":-/":         1,
	":-X":         1,
	":-]":         1,
	":-o":         1,
	":-p":         1,
	":-x":         1,
	":-|":         1,
	":-}":         1,
	":0":          1,
	":3":          1,
	":P":          1,
	":]":          1,
	":`(":         1,
	":`)":         1,
	":`-(":        1,
	":o":          1,
	":o)":         1,
	"=(":          1,
	"=)":          1,
	"=D":          1,
	"=|":          1,
	"@_@":         1,
	"O.o":         1,
	"O_o":         1,
	"V_V":         1,
	"XDD":         1,
	"[-:":         1,
	"^___^":       1,
	"o_0":         1,
	"o_O":         1,
	"o_o":         1,
	"v_v":         1,
	"xD":          1,
	"xDD":         1,
	"¯\\(ツ)/¯":    1,
}
