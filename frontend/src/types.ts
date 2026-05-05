import type { SearchResultItem } from './lib/api';

// === Search Result (re-export từ api.ts) ===
export type { SearchResultItem };

// === Navigation State types ===

/** State truyền từ Home → DiscoveryResults qua react-router */
export interface SearchNavigationState {
  results: SearchResultItem[];
  uploadedImage: string; // base64/blob data URL của ảnh user đã upload
  predicted_dish?: string;
  majority_votes?: number;
  vote_count?: number;
}

/** State truyền từ DiscoveryResults → DishDetail qua react-router */
export interface DishDetailNavigationState {
  dish_name: string;
  similarity: number;
  image_url: string;
  allResults: SearchResultItem[]; // tất cả kết quả để hiện "Món tương tự"
}

// === History (localStorage) ===

export interface HistoryItem {
  id: string;
  dish_name: string;
  similarity: number;
  image_url: string;       // ảnh kết quả từ backend
  uploadedImage: string;   // base64 ảnh user đã upload
  timestamp: number;       // Date.now()
  topResults: SearchResultItem[]; // top 5 kết quả
}

// === Legacy types (giữ lại nếu cần) ===

export interface RecipeStep {
  step: string;
  title: string;
  desc: string;
}

export interface Ingredient {
  name: string;
  icon: any;
  description: string;
}
