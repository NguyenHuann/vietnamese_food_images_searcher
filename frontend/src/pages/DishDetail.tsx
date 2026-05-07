import { Heart, MapPin, ArrowRight, ArrowLeft, Leaf, Utensils, Zap, Beef } from 'lucide-react';
import { motion } from 'motion/react';
import { Link, useLocation, Navigate } from 'react-router-dom';
import { getImageUrl } from '../lib/api';
import { useFavorites } from '../lib/useFavorites';
import type { DishDetailNavigationState, Ingredient, RecipeStep } from '../types';

import DISH_DATA from '../data/dishes.json';

const ICON_MAP: Record<string, any> = {
  Leaf, Utensils, Zap, Beef
};

const getIcon = (name: string) => ICON_MAP[name] || Utensils;

export default function DishDetail() {
  const location = useLocation();
  const state = location.state as DishDetailNavigationState | null;

  if (!state) {
    return <Navigate to="/" replace />;
  }

  const { dish_name, similarity, image_url, allResults } = state;
  const { isFavorite, toggleFavorite } = useFavorites();
  
  const imageFullUrl = getImageUrl(image_url);
  
  const dishInfo = (DISH_DATA as any)[dish_name];
  const ingredients = dishInfo?.ingredients || [];
  const recipeSteps = dishInfo?.recipe || [];
  const description = dishInfo?.description || `${dish_name} là một trong những món ăn truyền thống nổi tiếng của Việt Nam, mang đậm hương vị và bản sắc văn hóa ẩm thực dân tộc.`;

  const similarDishes = allResults.filter(r => r.image_url !== image_url).slice(0, 4);

  const favorited = isFavorite(dish_name);

  return (
    <div className="space-y-8 md:space-y-12 pb-24">
      <div>
        <button onClick={() => window.history.back()} className="flex items-center gap-2 text-on-surface-variant hover:text-primary transition-colors text-sm font-bold uppercase tracking-widest">
          <ArrowLeft size={16} /> Quay lại
        </button>
      </div>

      {/* Hero */}
      <section className="relative group">
        <div className="w-full h-[350px] md:h-[600px] rounded-[1.5rem] md:rounded-[2rem] overflow-hidden relative shadow-2xl bg-surface-container-highest">
          {/* Ảnh nền mờ để tạo chiều sâu */}
          <img 
            src={imageFullUrl} 
            alt="" 
            className="absolute inset-0 w-full h-full object-cover blur-2xl opacity-40 scale-110" 
          />
          {/* Ảnh chính hiển thị trọn vẹn */}
          <img 
            src={imageFullUrl} 
            alt={dish_name} 
            className="relative w-full h-full object-contain transition-transform duration-700 group-hover:scale-[1.02]" 
          />
          <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent" />
          <div className="absolute bottom-6 left-6 right-6 md:bottom-12 md:left-12 md:right-12 flex flex-col md:flex-row items-start md:items-end justify-between gap-6 md:gap-8">
            <div className="space-y-2 md:space-y-4">
              <span className="micro-label !text-white/60 !mb-0">Khám phá chi tiết</span>
              <h1 className="text-4xl md:text-8xl font-bold text-white tracking-tighter drop-shadow-sm leading-none">{dish_name}</h1>
              <span className={`inline-block px-4 py-1.5 rounded-full text-xs font-bold uppercase tracking-widest ${similarity >= 0.8 ? 'bg-green-500/80 text-white' : similarity >= 0.5 ? 'bg-amber-500/80 text-white' : 'bg-white/20 text-white'}`}>
                {(similarity * 100).toFixed(1)}% tương đồng
              </span>
            </div>
            <button 
              onClick={() => toggleFavorite({ dish_name, image_url, timestamp: Date.now() })}
              className={`flex items-center gap-3 px-6 md:px-10 py-3 md:py-5 font-bold rounded-full shadow-2xl active:scale-95 transition-all uppercase tracking-widest text-[10px] md:text-xs
                ${favorited 
                  ? 'bg-red-500 text-white hover:bg-red-600' 
                  : 'bg-white text-on-surface hover:bg-primary hover:text-white'
                }
              `}
            >
              <Heart size={18} className={favorited ? 'fill-current' : ''} />
              {favorited ? 'Đã lưu yêu thích' : 'Lưu vào yêu thích'}
            </button>
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 md:gap-16">
        <div className="lg:col-span-7 space-y-12 md:space-y-16">
          {/* Story */}
          <section className="space-y-6 md:space-y-8">
            <div className="space-y-2">
              <p className="micro-label">Câu chuyện di sản</p>
              <h2 className="text-3xl md:text-4xl font-bold italic tracking-tight">{dish_name}</h2>
            </div>
            <div className="space-y-4 md:space-y-6 text-lg md:text-xl text-on-surface-variant leading-relaxed font-body">
              <p className="italic">{description}</p>
              <p>Mỗi vùng miền lại có cách chế biến riêng, tạo nên sự đa dạng và phong phú cho nền ẩm thực Việt.</p>
            </div>
          </section>

          {/* Ingredients */}
          <section className="bg-surface-container-highest p-8 md:p-12 rounded-[2rem] md:rounded-[3rem] space-y-8 md:space-y-10">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              <div className="space-y-1">
                <p className="micro-label">Thành phần</p>
                <h2 className="text-3xl md:text-4xl font-bold italic tracking-tight">Nguyên liệu chính</h2>
              </div>
              <span className="flex items-center w-fit h-fit gap-2 px-6 py-2 bg-white text-primary rounded-full text-[10px] font-bold uppercase tracking-widest shadow-sm">
                <Leaf size={14} /> Tươi ngon & Bản địa
              </span>
            </div>
            {ingredients.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 md:gap-6">
                {ingredients.map((ing, i) => (
                  <motion.div key={ing.name} initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.05 }} className="relative group/tip">
                    <div className="bg-white p-6 md:p-8 rounded-2xl md:rounded-3xl border border-outline/5 text-center space-y-3 md:space-y-4 shadow-sm hover:shadow-xl transition-all h-full cursor-help">
                      <div className="w-12 h-12 md:w-16 md:h-16 bg-primary/5 rounded-full flex items-center justify-center mx-auto">
                        {(() => {
                          const Icon = getIcon(ing.icon);
                          return <Icon size={28} className="text-primary" />;
                        })()}
                      </div>
                      <p className="font-bold text-lg md:text-xl text-on-surface leading-snug">{ing.name}</p>
                    </div>
                    <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-4 w-48 opacity-0 group-hover/tip:opacity-100 pointer-events-none transition-all duration-300 transform translate-y-2 group-hover/tip:translate-y-0 z-50">
                      <div className="bg-on-surface text-surface p-4 rounded-2xl shadow-2xl text-xs md:text-sm leading-relaxed">
                        {ing.description}
                        <div className="absolute top-full left-1/2 -translate-x-1/2 border-8 border-transparent border-t-on-surface" />
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            ) : (
              <div className="py-12 text-center text-on-surface-variant/40 border-2 border-dashed border-white/50 rounded-2xl md:rounded-3xl font-serif italic">
                Thông tin nguyên liệu sẽ được cập nhật sớm.
              </div>
            )}
          </section>

          {/* Recipe */}
          <section className="bg-surface-container-low p-8 md:p-12 rounded-[2rem] md:rounded-[3rem] space-y-8 md:space-y-12 relative overflow-hidden">
            <div className="space-y-1 relative z-10">
              <p className="micro-label">Hướng dẫn</p>
              <h2 className="text-3xl md:text-4xl font-bold italic tracking-tight">Công thức chế biến</h2>
            </div>
            {recipeSteps.length > 0 ? (
              <motion.div className="space-y-12 md:space-y-16 relative" initial="hidden" whileInView="visible" viewport={{ once: true, margin: "-50px" }} variants={{ visible: { transition: { staggerChildren: 0.2 } } }}>
                <div className="absolute left-[1.75rem] md:left-[2.2rem] top-8 bottom-8 w-px bg-outline/10" />
                {recipeSteps.map((item) => (
                  <motion.div key={item.step} variants={{ hidden: { opacity: 0, x: -30 }, visible: { opacity: 1, x: 0, transition: { type: "spring", damping: 15, stiffness: 100 } } }} className="flex gap-6 md:gap-8 group relative z-10">
                    <div className="flex-shrink-0">
                      <motion.div whileHover={{ scale: 1.1 }} className="w-14 h-14 md:w-16 md:h-16 rounded-full bg-white border border-outline/5 shadow-sm flex items-center justify-center group-hover:bg-primary group-hover:text-white transition-all duration-500">
                        <span className="text-xl md:text-2xl font-serif font-bold italic">{item.step}</span>
                      </motion.div>
                    </div>
                    <div className="space-y-2 md:space-y-3 pt-1 md:pt-2">
                      <h4 className="text-xl md:text-2xl font-bold text-on-surface group-hover:text-primary transition-colors">{item.title}</h4>
                      <p className="text-base md:text-lg text-on-surface-variant leading-relaxed font-body max-w-2xl">{item.desc}</p>
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            ) : (
              <div className="py-24 text-center space-y-6 relative z-10">
                <div className="w-20 h-20 bg-surface-container rounded-full flex items-center justify-center mx-auto text-primary/10"><Utensils size={40} /></div>
                <div className="space-y-2">
                  <p className="text-xl font-serif italic text-on-surface-variant">Công thức chế biến sẽ được cập nhật sớm.</p>
                  <p className="text-sm text-on-surface-variant/60 font-medium uppercase tracking-widest">Vui lòng quay lại sau</p>
                </div>
              </div>
            )}
            <div className="absolute -top-12 -right-12 w-64 h-64 bg-primary/5 rounded-full blur-3xl" />
          </section>
        </div>

        {/* Right Column */}
        <div className="lg:col-span-5 space-y-16">
          {/* Similarity Stats */}
          <section className="bg-surface-container-low p-12 rounded-[3rem] space-y-10 border border-outline/5">
            <div className="space-y-1">
              <p className="micro-label">Phân tích AI</p>
              <h3 className="text-3xl font-bold italic tracking-tight">Kết quả nhận diện</h3>
            </div>
            <div className="space-y-10">
              <div className="flex justify-between items-end">
                <span className="text-on-surface-variant font-bold text-xl">Độ tương đồng</span>
                <span className="text-5xl font-bold text-on-surface leading-none">
                  {(similarity * 100).toFixed(1)} <span className="text-sm font-normal text-on-surface-variant uppercase tracking-widest ml-1">%</span>
                </span>
              </div>
              <div className="h-3 bg-surface-container-highest rounded-full overflow-hidden">
                <motion.div initial={{ width: 0 }} animate={{ width: `${similarity * 100}%` }} transition={{ duration: 1, ease: "easeOut" }} className={`h-full rounded-full ${similarity >= 0.8 ? 'bg-green-500' : similarity >= 0.5 ? 'bg-amber-500' : 'bg-red-400'}`} />
              </div>
              <p className={`text-center text-lg font-serif italic ${similarity >= 0.8 ? 'text-green-600' : similarity >= 0.5 ? 'text-amber-600' : 'text-red-500'}`}>
                {similarity >= 0.8 ? 'Rất chính xác' : similarity >= 0.5 ? 'Khá chính xác' : 'Độ chính xác thấp'}
              </p>
            </div>
          </section>

          {/* Similar Dishes */}
          {similarDishes.length > 0 && (
            <section className="bg-surface-container-highest rounded-[3rem] overflow-hidden border border-outline/10 p-8 md:p-12 space-y-8">
              <div className="space-y-1">
                <p className="micro-label">Gợi ý</p>
                <h3 className="text-3xl font-bold italic tracking-tight">Món tương tự</h3>
              </div>
              <div className="space-y-4">
                {similarDishes.map((dish, i) => (
                  <Link key={`${dish.image_url}-${i}`} to="/detail" state={{ dish_name: dish.dish_name, similarity: dish.similarity, image_url: dish.image_url, allResults }} className="flex items-center gap-4 p-3 rounded-2xl hover:bg-white/80 transition-all group">
                    <div className={`w-16 h-16 rounded-xl overflow-hidden flex-shrink-0 shadow-sm border-2
                      ${dish.is_correct === true ? 'border-green-500' : dish.is_correct === false ? 'border-red-500' : 'border-transparent'}
                    `}>
                      <img src={getImageUrl(dish.image_url)} alt={dish.dish_name} className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-500" loading="lazy" />
                    </div>
                    <div className="flex-grow min-w-0">
                      <p className="font-bold text-on-surface group-hover:text-primary transition-colors truncate">{dish.dish_name}</p>
                      <p className="text-sm text-on-surface-variant">{(dish.similarity * 100).toFixed(1)}%</p>
                    </div>
                    <ArrowRight size={16} className="text-on-surface-variant/30 group-hover:text-primary transition-colors flex-shrink-0" />
                  </Link>
                ))}
              </div>
            </section>
          )}

          {/* Map */}
          <section className="flex flex-col h-fit bg-surface-container-highest rounded-[3rem] overflow-hidden shadow-2xl group border border-outline/10">
            <div className="h-48 relative bg-gradient-to-br from-primary/10 to-primary/5 overflow-hidden flex items-center justify-center">
              <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center shadow-2xl">
                <MapPin size={32} className="text-primary" />
              </div>
            </div>
            <div className="p-12 space-y-8">
              <div className="space-y-4">
                <h4 className="text-3xl font-bold italic tracking-tight">Tìm quán {dish_name}</h4>
                <p className="text-on-surface-variant text-lg leading-relaxed font-body">Tìm kiếm các địa điểm phục vụ {dish_name} quanh khu vực của bạn.</p>
              </div>
              <button 
                onClick={() => window.open(`https://www.google.com/search?q=quán+${encodeURIComponent(dish_name)}+gần+đây`, '_blank')}
                className="w-full bg-on-surface text-white py-6 rounded-full font-bold text-xs uppercase tracking-[0.3em] shadow-xl hover:bg-primary transition-all active:scale-95 flex items-center justify-center gap-4"
              >
                Tìm nhà hàng gần đây <ArrowRight size={20} />
              </button>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
