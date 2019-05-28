#include <pebble.h>
#include <stdlib.h>

static Window *s_main_window;

static GBitmap *s_bitmap = NULL;
static Layer *window_layer;
static BitmapLayer *s_bitmap_layer;
static TextLayer *s_text_layer, *s_loss_layer;
static GBitmapSequence *s_sequence = NULL;

#define NUMPAT 4
#define NUMIN  2
#define NUMHID 2
#define NUMOUT 1

#define rando() ((double)rand()/((double)RAND_MAX+1))

// Taylor Series approximation for exp(x)
double exp(double x) {
    double sum = 1.0;

    for(int i = 10; i > 0; --i)
        sum = 1 + x * sum / i;

    return sum;
}

char* concat(const char *s1, const char *s2){
    char *catstr = malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(catstr, s1);
    strcat(catstr, s2);
    return catstr;
}

static void create_loss_layer() {
    s_loss_layer = text_layer_create(PBL_IF_ROUND_ELSE(
       GRect(bounds.origin.x, 150, bounds.size.w, 100),
       GRect(8, 150, 136, 100)));
     text_layer_set_text_alignment(s_loss_layer, PBL_IF_ROUND_ELSE(
       GTextAlignmentCenter, GTextAlignmentCenter));
     text_layer_set_text_color(s_loss_layer, GColorRed);
     text_layer_set_background_color(s_loss_layer, GColorClear);
     text_layer_set_font(s_loss_layer, fonts_get_system_font(FONT_KEY_GOTHIC_18_BOLD));

     layer_add_child(window_layer, text_layer_get_layer(s_loss_layer));
}

// Adapted from John Bullinaria
// http://www.cs.bham.ac.uk/~jxb/INC/nn.c
static void train_mlp() {
    int    i, j, k, p, np, op, ranpat[NUMPAT+1], epoch;
    int    NumPattern = NUMPAT, NumInput = NUMIN, NumHidden = NUMHID, NumOutput = NUMOUT;
    double Input[NUMPAT+1][NUMIN+1] = { {0, 0, 0},  {0, 0, 0},  {0, 1, 0},  {0, 0, 1},  {0, 1, 1} };
    double Target[NUMPAT+1][NUMOUT+1] = { {0, 0},  {0, 0},  {0, 1},  {0, 1},  {0, 0} };
    double SumH[NUMPAT+1][NUMHID+1], WeightIH[NUMIN+1][NUMHID+1], Hidden[NUMPAT+1][NUMHID+1];
    double SumO[NUMPAT+1][NUMOUT+1], WeightHO[NUMHID+1][NUMOUT+1], Output[NUMPAT+1][NUMOUT+1];
    double DeltaO[NUMOUT+1], SumDOW[NUMHID+1], DeltaH[NUMHID+1];
    double DeltaWeightIH[NUMIN+1][NUMHID+1], DeltaWeightHO[NUMHID+1][NUMOUT+1];
    double Error, eta = 0.5, alpha = 0.9, smallwt = 0.5;

    for( j = 1 ; j <= NumHidden ; j++ ) {    /* initialize WeightIH and DeltaWeightIH */
        for( i = 0 ; i <= NumInput ; i++ ) {
            DeltaWeightIH[i][j] = 0.0 ;
            WeightIH[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }
    for( k = 1 ; k <= NumOutput ; k ++ ) {    /* initialize WeightHO and DeltaWeightHO */
        for( j = 0 ; j <= NumHidden ; j++ ) {
            DeltaWeightHO[j][k] = 0.0 ;
            WeightHO[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }

    for( epoch = 0 ; epoch < 1000 ; epoch++) {    /* iterate weight updates */
        for( p = 1 ; p <= NumPattern ; p++ ) {    /* randomize order of training patterns */
            ranpat[p] = p ;
        }
        for( p = 1 ; p <= NumPattern ; p++) {
            np = p + rando() * ( NumPattern + 1 - p ) ;
            op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
        }
        Error = 0.0 ;
        for( np = 1 ; np <= NumPattern ; np++ ) {    /* repeat for all the training patterns */
            p = ranpat[np];
            for( j = 1 ; j <= NumHidden ; j++ ) {    /* compute hidden unit activations */
                SumH[p][j] = WeightIH[0][j] ;
                for( i = 1 ; i <= NumInput ; i++ ) {
                    SumH[p][j] += Input[p][i] * WeightIH[i][j] ;
                }
                Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j])) ;
            }
            for( k = 1 ; k <= NumOutput ; k++ ) {    /* compute output unit activations and errors */
                SumO[p][k] = WeightHO[0][k] ;
                for( j = 1 ; j <= NumHidden ; j++ ) {
                    SumO[p][k] += Hidden[p][j] * WeightHO[j][k] ;
                }
                Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k])) ;   /* Sigmoidal Outputs */
/*              Output[p][k] = SumO[p][k];      Linear Outputs */
                Error += 0.5 * (Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k]) ;   /* SSE */
/*              Error -= ( Target[p][k] * log( Output[p][k] ) + ( 1.0 - Target[p][k] ) * log( 1.0 - Output[p][k] ) ) ;    Cross-Entropy Error */
                DeltaO[k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;   /* Sigmoidal Outputs, SSE */
/*              DeltaO[k] = Target[p][k] - Output[p][k];     Sigmoidal Outputs, Cross-Entropy Error */
/*              DeltaO[k] = Target[p][k] - Output[p][k];     Linear Outputs, SSE */
            }
            for( j = 1 ; j <= NumHidden ; j++ ) {    /* backpropagate errors to hidden layer */
                SumDOW[j] = 0.0 ;
                for( k = 1 ; k <= NumOutput ; k++ ) {
                    SumDOW[j] += WeightHO[j][k] * DeltaO[k] ;
                }
                DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
            }
            for( j = 1 ; j <= NumHidden ; j++ ) {     /* update weights WeightIH */
                DeltaWeightIH[0][j] = eta * DeltaH[j] + alpha * DeltaWeightIH[0][j] ;
                WeightIH[0][j] += DeltaWeightIH[0][j] ;
                for( i = 1 ; i <= NumInput ; i++ ) {
                    DeltaWeightIH[i][j] = eta * Input[p][i] * DeltaH[j] + alpha * DeltaWeightIH[i][j];
                    WeightIH[i][j] += DeltaWeightIH[i][j] ;
                }
            }
            for(k = 1; k <= NumOutput; k++) {    /* update weights WeightHO */
                DeltaWeightHO[0][k] = eta * DeltaO[k] + alpha * DeltaWeightHO[0][k];
                WeightHO[0][k] += DeltaWeightHO[0][k];
                for(j = 1; j <= NumHidden; j++) {
                    DeltaWeightHO[j][k] = eta * Hidden[p][j] * DeltaO[k] + alpha * DeltaWeightHO[j][k];
                    WeightHO[j][k] += DeltaWeightHO[j][k];
                }
            }
        }

        int len = snprintf(NULL, 0, "%i", epoch);
        char *epochstr = malloc(len+1);
        snprintf(epochstr, len, "%i", epoch);
        int len2 = snprintf(NULL, 0, "%f", Error);
        char *errorstr = malloc(len2+1);
        snprintf(errorstr, len2, "%f", Error);

        char *tmp = concat("Epoch: ", epochstr);
        char *tmp2 = concat(" Loss: ", errorstr);
        char *labeltxt = concat(tmp, tmp2);

        text_layer_set_text(s_loss_layer, labeltxt);
        free(labeltxt);
        free(tmp);
        free(tmp2);
        free(epochstr);
        free(errorstr);

        // stop learning, loss has decreased enough
        if(Error < 0.0004){
            int len = snprintf(NULL, 0, "%f", Error);
            char *errorstr = malloc(len+1);
            snprintf(errorstr, len, "%f", Error);
            char *labeltxt = concat("DONE. Loss: ", errorstr);

            text_layer_set_text(s_loss_layer, labeltxt);
            free(errorstr);
            break;
        }
    }
}

static void load_sequence();

static void timer_handler(void *context) {
  uint32_t next_delay;

  // Advance to the next APNG frame
  if(gbitmap_sequence_update_bitmap_next_frame(s_sequence, s_bitmap, &next_delay)) {
    bitmap_layer_set_bitmap(s_bitmap_layer, s_bitmap);
    layer_mark_dirty(bitmap_layer_get_layer(s_bitmap_layer));

    // Timer for that delay
    app_timer_register(next_delay, timer_handler, NULL);
  } else {
    // Start again
    load_sequence();
  }
}

static void load_sequence() {
  // Free old data
  if(s_sequence) {
    gbitmap_sequence_destroy(s_sequence);
    s_sequence = NULL;
  }
  if(s_bitmap) {
    gbitmap_destroy(s_bitmap);
    s_bitmap = NULL;
  }

  // Create sequence
  s_sequence = gbitmap_sequence_create_with_resource(RESOURCE_ID_ANIMATION);

  // Create GBitmap
  s_bitmap = gbitmap_create_blank(gbitmap_sequence_get_bitmap_size(s_sequence), GBitmapFormat8Bit);

  // Begin animation
  app_timer_register(1, timer_handler, NULL);
}

static void main_window_load(Window *window) {
  window_layer = window_get_root_layer(window);
  GRect bounds = layer_get_bounds(window_layer);

  s_bitmap_layer = bitmap_layer_create(bounds);
  layer_add_child(window_layer, bitmap_layer_get_layer(s_bitmap_layer));

  s_text_layer = text_layer_create(PBL_IF_ROUND_ELSE(
    GRect(bounds.origin.x, 2, bounds.size.w, 100),
    GRect(8, 2, 136, 100)));
  text_layer_set_text_alignment(s_text_layer, PBL_IF_ROUND_ELSE(
    GTextAlignmentCenter, GTextAlignmentCenter));
  text_layer_set_text_color(s_text_layer, GColorGreen);
  text_layer_set_background_color(s_text_layer, GColorClear);
  text_layer_set_font(s_text_layer, fonts_get_system_font(FONT_KEY_GOTHIC_18_BOLD));

  layer_add_child(window_layer, text_layer_get_layer(s_text_layer));

  text_layer_set_text(s_text_layer, "GET CONVOLUTIONAL");

  create_loss_layer();

  load_sequence();

  train_mlp();
}

static void main_window_unload(Window *window) {
  text_layer_destroy(s_loss_layer);
  text_layer_destroy(s_text_layer);
  bitmap_layer_destroy(s_bitmap_layer);
}

static void init() {
  s_main_window = window_create();
  window_set_background_color(s_main_window, GColorWhite);
#if defined(PBL_SDK_2)
  window_set_fullscreen(s_main_window, true);
#endif
  window_set_window_handlers(s_main_window, (WindowHandlers) {
    .load = main_window_load,
    .unload = main_window_unload,
  });
  window_stack_push(s_main_window, true);
}

static void deinit() {
  window_destroy(s_main_window);
}

int main() {
  init();
  app_event_loop();
  deinit();
}
