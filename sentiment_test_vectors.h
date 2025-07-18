#ifndef SENTIMENT_TEST_VECTORS_H
#define SENTIMENT_TEST_VECTORS_H

#define NUM_CLASSES 5
#define NUM_INPUTS 133
#define NUM_SAMPLES 45

typedef struct {
    const float* input;
    const float* expected;
    const char* text;
} SentimentSample;

extern const float review_input_0[5][133];
extern const float review_input_1[5][133];
extern const float review_input_2[5][133];
extern const float review_input_3[5][133];
extern const float review_input_4[5][133];

extern const float review_output_0[5][5];
extern const float review_output_1[5][5];
extern const float review_output_2[5][5];
extern const float review_output_3[5][5];
extern const float review_output_4[5][5];

extern const char* review_text_0[5];
extern const char* review_text_1[5];
extern const char* review_text_2[5];
extern const char* review_text_3[5];
extern const char* review_text_4[5];

extern SentimentSample test_cases[45];

extern void init_test_cases(void);
#endif // SENTIMENT_TEST_VECTORS_H



