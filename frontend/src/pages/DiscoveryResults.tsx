import { useState } from 'react';
import { Sparkles, Upload, Info, ChevronDown, CheckCircle2, AlertTriangle } from 'lucide-react';
import { motion } from 'motion/react';
import { Link, useLocation, Navigate } from 'react-router-dom';
import { getImageUrl } from '../lib/api';
import type { SearchNavigationState, DishDetailNavigationState, SearchResultItem } from '../types';

const ITEMS_PER_PAGE = 12;

export default function DiscoveryResults() {
  const location = useLocation();
  const state = location.state as SearchNavigationState | null;
  const [visibleCount, setVisibleCount] = useState(ITEMS_PER_PAGE);

  // Nếu không có state (user vào trực tiếp URL) → redirect về Home
  if (!state || !state.results) {
    return <Navigate to="/" replace />;
  }

  const { results, uploadedImage, predicted_dish, majority_votes, vote_count } = state;
  const visibleResults = results.slice(0, visibleCount);
  const hasMore = visibleCount < results.length;

  const top1Score = results.length > 0 ? results[0].similarity : 0;
  const isMatchFound = top1Score >= 0.70;

  return (
    <div className="space-y-12">
      {/* Analysis Header */}
      <section className="bg-surface-container-low rounded-3xl p-6 md:p-8 flex flex-col md:flex-row items-center gap-8 border border-outline-variant/5">
        <div className="relative group">
          <div className="w-32 h-32 md:w-40 md:h-40 rounded-2xl overflow-hidden shadow-sm border-4 border-surface ring-1 ring-outline/5 transition-transform duration-500 group-hover:scale-105">
            <img 
              src={uploadedImage} 
              alt="Ảnh đã tải lên" 
              className="w-full h-full object-cover"
              referrerPolicy="no-referrer"
            />
          </div>
          <span className="absolute -top-2 -right-2 bg-primary text-on-primary text-[10px] font-bold px-3 py-1 rounded-full uppercase tracking-widest shadow-lg">
            Gốc
          </span>
        </div>
        
        <div className="text-center md:text-left space-y-2">
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight">
            {predicted_dish 
              ? `Dự đoán: ${predicted_dish}`
              : results.length > 0 ? `Nhận diện: ${results[0].dish_name}` : 'Không tìm thấy kết quả'
            }
          </h1>
          <p className="text-on-surface-variant max-w-xl text-lg leading-relaxed font-serif italic">
            {results.length > 0 
              ? `Độ chính xác: ${(results[0].similarity * 100).toFixed(1)}% — ${vote_count ? `Hệ thống bầu chọn: ${majority_votes}/${vote_count} phiếu` : `Tìm thấy ${results.length} kết quả`}.`
              : 'Hãy thử tải lên ảnh khác để AI nhận diện.'
            }
          </p>
        </div>
      </section>

      {/* Match Banner */}
      {results.length > 0 && (
        <section>
          {isMatchFound ? (
            <div className="bg-green-500/10 border-l-4 border-green-500 p-6 md:p-8 rounded-2xl shadow-sm flex items-center gap-6 transition-all">
              <CheckCircle2 size={40} className="text-green-600 flex-shrink-0" />
              <div>
                <h2 className="text-xl font-bold text-green-800">Tìm thấy hình ảnh phù hợp</h2>
                <p className="text-green-700 mt-1 font-medium">Dưới đây là các kết quả có độ tương đồng cao với ảnh của bạn. Phân tích AI được cho là có độ tin cậy cao.</p>
              </div>
            </div>
          ) : (
            <div className="bg-red-500/10 border-l-4 border-red-500 p-6 md:p-8 rounded-2xl shadow-sm flex items-center gap-6 transition-all">
              <AlertTriangle size={40} className="text-red-600 flex-shrink-0" />
              <div>
                <h2 className="text-xl font-bold text-red-800">Không tìm thấy ảnh phù hợp (Độ tin cậy thấp)</h2>
                <p className="text-red-700 mt-1 font-medium">Đây là những hình ảnh có nét tương tự với ảnh của bạn. Tuy nhiên, AI không đủ tự tin để đưa ra kết luận chính xác.</p>
              </div>
            </div>
          )}
        </section>
      )}

      {/* Results Section */}
      <section className="space-y-12">
        <div className="flex flex-col items-center text-center space-y-4 border-b border-outline/10 pb-8">
          <p className="micro-label">Kết quả phân tích</p>
          <h2 className="text-4xl font-bold flex items-center gap-3">
            <Sparkles size={28} className="text-primary" />
            Các món ăn tương tự
          </h2>
          <span className="text-xs font-bold bg-surface-container-highest px-6 py-2 rounded-full uppercase tracking-[0.2em] text-on-surface-variant">
            {results.length} Kết quả được tìm thấy
          </span>
        </div>

        {visibleResults.length > 0 ? (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-12">
              {visibleResults.map((item, index) => {
                const detailState: DishDetailNavigationState = {
                  dish_name: item.dish_name,
                  similarity: item.similarity,
                  image_url: item.image_url,
                  allResults: results,
                };

                return (
                  <motion.div 
                    key={`${item.image_url}-${index}`}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: Math.min(index * 0.05, 0.5) }}
                    className="group"
                  >
                    <Link to="/detail" state={detailState} className="block space-y-6 text-center">
                      <div className={`relative aspect-[4/5] rounded-[2.5rem] overflow-hidden bg-surface-container-low shadow-xl group-hover:shadow-2xl transition-all duration-700
                        ${isMatchFound && item.is_correct === true ? 'ring-8 ring-green-500 ring-inset' : isMatchFound && item.is_correct === false ? 'ring-8 ring-red-500 ring-inset' : 'border border-outline/10'}
                      `}>
                        <img 
                          src={getImageUrl(item.image_url)} 
                          alt={item.dish_name} 
                          className="w-full h-full object-cover transition-transform duration-1000 group-hover:scale-110"
                          referrerPolicy="no-referrer"
                          loading="lazy"
                        />
                        <div className="absolute top-6 right-6 flex flex-col gap-2 items-end">
                          <span className={`px-5 py-2 text-[10px] font-bold uppercase tracking-widest rounded-full shadow-2xl backdrop-blur-sm
                            ${item.similarity >= 0.8 
                              ? 'bg-primary text-on-primary' 
                              : item.similarity >= 0.5 
                                ? 'bg-amber-500 text-white' 
                                : 'bg-surface-container-highest text-on-surface'
                            }
                          `}>
                            {(item.similarity * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                      
                      <div className="space-y-4 px-4">
                        <div className="space-y-2">
                          <h3 className="text-2xl font-bold text-on-surface group-hover:text-primary transition-colors">
                            {item.dish_name}
                          </h3>
                          <p className="text-sm text-on-surface-variant font-serif italic line-clamp-2 leading-relaxed">
                            Độ tương đồng: {(item.similarity * 100).toFixed(1)}%
                          </p>
                        </div>
                      </div>
                    </Link>
                  </motion.div>
                );
              })}
            </div>

            {/* Load More Button */}
            {hasMore && (
              <div className="flex justify-center pt-4">
                <button
                  onClick={() => setVisibleCount(prev => prev + ITEMS_PER_PAGE)}
                  className="flex items-center gap-3 px-10 py-4 border border-outline/20 rounded-full font-bold uppercase text-[10px] tracking-[0.2em] hover:bg-on-surface hover:text-white transition-all shadow-sm hover:shadow-xl group"
                >
                  <span>Xem thêm ({results.length - visibleCount} còn lại)</span>
                  <ChevronDown size={16} className="group-hover:translate-y-0.5 transition-transform" />
                </button>
              </div>
            )}
          </>
        ) : (
          <div className="py-24 text-center space-y-4 border-2 border-dashed border-outline/10 rounded-[3rem] bg-surface-container-lowest">
            <div className="inline-flex w-20 h-20 bg-surface-container rounded-full items-center justify-center text-primary/20">
              <Sparkles size={40} />
            </div>
            <p className="text-xl font-serif italic text-on-surface-variant">Không tìm thấy món ăn tương tự. Hãy thử ảnh khác.</p>
          </div>
        )}
      </section>

      {/* Footer CTA */}
      <section className="pt-12 flex flex-col items-center gap-8">
        <p className="text-on-surface-variant font-medium flex items-center gap-2">
          <Info size={18} />
          Không phải món bạn đang tìm?
        </p>
        <Link 
          to="/"
          className="bg-gradient-to-r from-primary to-primary-container text-on-primary px-8 py-4 rounded-2xl font-bold text-lg flex items-center gap-3 shadow-xl hover:scale-105 active:scale-95 transition-all"
        >
          <Upload size={20} />
          Tải lên ảnh khác
        </Link>
      </section>
    </div>
  );
}
