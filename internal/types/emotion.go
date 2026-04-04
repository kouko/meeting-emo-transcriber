package types

type EmotionInfo struct {
	Raw     string `json:"raw"`
	Label   string `json:"label"`
	Display string `json:"display"`
}

type EmotionResult struct {
	Raw        string
	Label      string
	Display    string
	Confidence float32
}

type EmotionMapping struct {
	Label   string
	Display string
}

var SenseVoiceEmotionMap = map[string]EmotionMapping{
	"HAPPY":     {Label: "Happy", Display: "happily"},
	"SAD":       {Label: "Sad", Display: "sadly"},
	"ANGRY":     {Label: "Angry", Display: "angrily"},
	"NEUTRAL":   {Label: "Neutral", Display: ""},
	"FEARFUL":   {Label: "Fearful", Display: "fearfully"},
	"DISGUSTED": {Label: "Disgusted", Display: "with disgust"},
	"SURPRISED": {Label: "Surprised", Display: "with surprise"},
	"unk":       {Label: "Unknown", Display: ""},
}

var Emotion2vecEmotionMap = map[string]EmotionMapping{
	"angry":     {Label: "Angry", Display: "angrily"},
	"disgusted": {Label: "Disgusted", Display: "with disgust"},
	"fearful":   {Label: "Fearful", Display: "fearfully"},
	"happy":     {Label: "Happy", Display: "happily"},
	"neutral":   {Label: "Neutral", Display: ""},
	"other":     {Label: "Other", Display: ""},
	"sad":       {Label: "Sad", Display: "sadly"},
	"surprised": {Label: "Surprised", Display: "with surprise"},
	"unknown":   {Label: "Unknown", Display: ""},
}

var AudioEventDisplayMap = map[string]string{
	"Speech":   "",
	"BGM":      "[background music]",
	"Applause": "[applause]",
	"Laughter": "[laughter]",
	"Cry":      "[crying]",
	"Sneeze":   "[sneeze]",
	"Breath":   "[breathing]",
	"Cough":    "[cough]",
}

func LookupEmotion(raw string, mapping map[string]EmotionMapping) EmotionInfo {
	if m, ok := mapping[raw]; ok {
		return EmotionInfo{Raw: raw, Label: m.Label, Display: m.Display}
	}
	return EmotionInfo{Raw: raw, Label: "Unknown", Display: ""}
}
