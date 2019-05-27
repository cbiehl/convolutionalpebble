#include <pebble.h>

static Window *s_main_window;

static GBitmap *s_bitmap = NULL;
static BitmapLayer *s_bitmap_layer;
static TextLayer *s_text_layer;
static GBitmapSequence *s_sequence = NULL;

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
  Layer *window_layer = window_get_root_layer(window);
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

  load_sequence();
}

static void main_window_unload(Window *window) {
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