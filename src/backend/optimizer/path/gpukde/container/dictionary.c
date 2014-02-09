// Inhibit gcc warnings on mixed declarations
#pragma GCC diagnostic ignored "-Wdeclaration-after-statement"

#include "dictionary.h"

#include <stdlib.h>
#include <string.h>

// Helper function to hash strings 
// (Uses Austin Appleby's MurmurHash implementation)
static unsigned int MurmurHash2(const char* key, unsigned int len) {
   unsigned int seed = 0xDEADBEEF;
   
	// 'm' and 'r' are mixing constants generated offline.
	// They're not really 'magic', they just happen to work well.
	const unsigned int m = 0x5bd1e995;
	const int r = 24;

	// Initialize the hash to a 'random' value
	unsigned int h = seed ^ len;

	// Mix 4 bytes at a time into the hash
	const unsigned char * data = (const unsigned char *)key;
	while(len >= 4) {
		unsigned int k = *(unsigned int *)data;
		k *= m; 
		k ^= k >> r; 
		k *= m; 
		h *= m; 
		h ^= k;
      
		data += 4;
		len -= 4;
	}
	
	// Handle the last few bytes of the input array
	switch(len) {
	case 3: h ^= data[2] << 16;
	case 2: h ^= data[1] << 8;
	case 1: h ^= data[0];
	        h *= m;
	};

	// Do a few final mixes of the hash to ensure the last few
	// bytes are well-incorporated.
	h ^= h >> 13;
	h *= m;
	h ^= h >> 15;
	return h;
} 


dictionary_t dictionary_init(void) {
   dictionary_t new_dict = (dictionary_t)calloc(1, sizeof(dictionary_tt));
   // Allocate the hash table.
   new_dict->hash_table_size = 1024;
   new_dict->hash_table = (dictionary_entry_t**)calloc(
           1, new_dict->hash_table_size * sizeof(dictionary_entry_t*));
   return new_dict;
}


void dictionary_release(dictionary_t dictionary, char release_payloads) {
   if (!dictionary) return;
   unsigned int i;
   for (i = 0; i < dictionary->hash_table_size; ++i) {
      dictionary_entry_t* entry = dictionary->hash_table[i];
      while (entry) {
         free(entry->key);
         if (release_payloads) free(entry->payload);
         dictionary_entry_t* next_entry = entry->next;
         free(entry);
         entry = next_entry;
      }
   }
   free(dictionary->hash_table);
   free(dictionary);
}


void* dictionary_get(dictionary_t dictionary, const char* key) {
   unsigned int len = strlen(key);
   // Compute the hash bucket for the string.
   unsigned int bucket = MurmurHash2(key, len) % dictionary->hash_table_size;
   // Check whether the key is already contained in the bucket.
   dictionary_entry_t* entry = dictionary->hash_table[bucket];
   while (entry) {
      if (strcmp(key, entry->key) == 0) break;
      entry = entry->next;
   }
   if (entry) return entry->payload;
   else return NULL;
}


void* dictionary_insert(dictionary_t dictionary, const char* key, void* payload) {
   unsigned int len = strlen(key);
   // Compute the hash bucket for the string.
   unsigned int bucket = MurmurHash2(key, len) % dictionary->hash_table_size;
   // Check whether the key is already contained in the bucket.
   dictionary_entry_t* entry = dictionary->hash_table[bucket];
   while (entry) {
      if (strcmp(key, entry->key) == 0) break;
      entry = entry->next;
   }
   if (entry) {
      // We already know this key, simply update the payload.
      void* tmp = entry->payload;
      entry->payload = payload;
      return tmp;
   } else {
      // Allocate a new entry and insert it in the bucket.
      entry = (dictionary_entry_t*)calloc(1, sizeof(dictionary_entry_t));
      entry->next = dictionary->hash_table[bucket];
      dictionary->hash_table[bucket] = entry;
      // Copy the key.
      entry->key = malloc(len + 1);
      strcpy(entry->key, key);
      // And set the payload.
      entry->payload = payload;
      return NULL;
   }
}

void dictionary_remove(dictionary_t dictionary, const char* key, char release_payload) {
   unsigned int len = strlen(key);
   // Compute the hash bucket for the string.
   unsigned int bucket = MurmurHash2(key, len) % dictionary->hash_table_size;
   // Find the pointer that points to the element that should be deleted.
   dictionary_entry_t** next_ptr = &(dictionary->hash_table[bucket]);
   while (*next_ptr && strcmp((*next_ptr)->key, key) != 0) {
      next_ptr = &((*next_ptr)->next);
   }
   if (*next_ptr) {
      dictionary_entry_t* entry = *next_ptr;
      // Remove the entry from the list.
      *next_ptr = entry->next;
      // Now delete the element.
      free(entry->key);
      if (release_payload) free(entry->payload);
      free(entry);
   }
}

// Iterator functionality.

dictionary_iterator_t dictionary_iterator_init(dictionary_t dictionary) {
  dictionary_iterator_t tmp;
  tmp.current_bucket = 0;
  tmp.current_list_element = NULL;
  tmp = dictionary_iterator_next(dictionary, tmp);
  return tmp;
}

dictionary_iterator_t dictionary_iterator_next(dictionary_t dictionary,
                                               dictionary_iterator_t iterator) {
  if (dictionary == NULL) {
    dictionary_iterator_t empty_iterator;
    empty_iterator.current_bucket = 0;
    empty_iterator.current_list_element = NULL;
    return empty_iterator;
  }
  // Check if we are currently within a valid bucket.
  if (iterator.current_list_element) {
    iterator.current_list_element = iterator.current_list_element->next;
    if (iterator.current_list_element) return iterator;
    iterator.current_list_element = NULL;
    iterator.current_bucket++;
  }
  // Walk the buckets until we either find a non-empty bucket, or reach the end.
  unsigned int bucket = iterator.current_bucket;
  for (; bucket < dictionary->hash_table_size; ++bucket) {
    if (dictionary->hash_table[bucket]) {
      iterator.current_list_element = dictionary->hash_table[bucket];
      iterator.current_bucket = bucket;
      break;
    }
  }
  return iterator;
}

const char* dictionary_iterator_key(dictionary_t dictionary,
                                    dictionary_iterator_t iterator) {
  if (dictionary == NULL) return NULL;
  if (iterator.current_list_element == NULL) return NULL;
  return iterator.current_list_element->key;
}

void* dictionary_iterator_value(dictionary_t dictionary,
                                dictionary_iterator_t iterator) {
  if (dictionary == NULL) return NULL;
  if (iterator.current_list_element == NULL) return NULL;
  return iterator.current_list_element->payload;
}
