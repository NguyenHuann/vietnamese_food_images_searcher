import React, { useState, useRef } from 'react';
import { Camera, ChevronRight, Verified, Sparkles, X, UploadCloud, Loader2, Settings2 } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { Link, useNavigate } from 'react-router-dom';
import { searchByImage } from '../lib/api';
import { useSearchHistory } from '../lib/useSearchHistory';
import { compressImage } from '../lib/utils';
import type { SearchNavigationState, HistoryItem } from '../types';

export default function Home() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [kValue, setKValue] = useState<number>(10);
  const [loadingStep, setLoadingStep] = useState<string>('Đang khởi tạo...');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const { history, addSearch } = useSearchHistory();

  // Quản lý Object URL để tránh rò rỉ bộ nhớ
  React.useEffect(() => {
    return () => {
      if (selectedImage && selectedImage.startsWith('blob:')) {
        URL.revokeObjectURL(selectedImage);
      }
    };
  }, [selectedImage]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
  };

  const processFile = (file: File) => {
    setSelectedFile(file);
    setError(null);
    
    // Thu hồi URL cũ nếu có
    if (selectedImage && selectedImage.startsWith('blob:')) {
      URL.revokeObjectURL(selectedImage);
    }
    
    // Tạo URL mới cực ngắn (blob:) thay vì Base64 cực dài
    const objectUrl = URL.createObjectURL(file);
    setSelectedImage(objectUrl);
  };

  const [isDragging, setIsDragging] = React.useState(false);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith('image/')) {
      processFile(file);
    } else {
      setError('Vui lòng chỉ tải lên tệp tin hình ảnh.');
    }
  };

  const handleUploadClick = () => {
    if (!isSearching) {
      fileInputRef.current?.click();
    }
  };

  const handleSearch = async () => {
    if (!selectedFile || !selectedImage) return;

    setIsSearching(true);
    setError(null);

    try {
      setLoadingStep('Đang gửi ảnh lên Server...');
      const data = await searchByImage(selectedFile, kValue, (step) => {
        setLoadingStep(step);
      });
      
      if (data.error) {
        setError(data.error);
        return;
      }

      // 1. Tạo thumbnail nhỏ để lưu lịch sử và truyền qua State (tránh lỗi revoke URL khi unmount)
      const thumbnail = await compressImage(selectedImage, 400);
      await addSearch(thumbnail, data.results);

      // 2. Navigate đến trang kết quả
      const state: SearchNavigationState = {
        results: data.results,
        uploadedImage: thumbnail,
        predicted_dish: data.predicted_dish,
        majority_votes: data.majority_votes,
        vote_count: data.vote_count,
      };
      navigate('/results', { state });
    } catch (err: any) {
      setError(err.message || 'Không thể kết nối đến server. Hãy kiểm tra backend đang chạy.');
    } finally {
      setIsSearching(false);
    }
  };

  // Lấy 6 kết quả gần đây nhất từ history để hiển thị
  const recentItems = history.slice(0, 6);

  return (
    <div className="space-y-16">
      {/* Hero */}
      <section className="space-y-4 pt-12">
        <div className="flex flex-col items-center text-center space-y-6">
          <div className="space-y-1">
            <p className="micro-label">Khám phá di sản</p>
            <motion.h1 
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="text-5xl md:text-7xl font-bold leading-[1.1] text-on-surface"
            >
              Hệ Thống Tra Cứu Món Ăn Việt Nam
            </motion.h1>
            <p className="text-xl md:text-2xl font-serif italic text-primary">Tinh hoa hội tụ, đậm đà bản sắc</p>
          </div>
          <motion.p 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="text-on-surface-variant max-w-2xl text-lg leading-relaxed"
          >
            Nhận diện món ăn chính gốc, khám phá công thức gia truyền và tôn vinh những giá trị văn hóa ngàn năm.
          </motion.p>
        </div>
      </section>

      {/* Upload Zone */}
      <motion.section 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="relative"
      >
        <input 
          type="file" 
          ref={fileInputRef} 
          className="hidden" 
          accept="image/*" 
          onChange={handleFileChange} 
        />
        
        <div className="relative group">
          <div 
            onClick={handleUploadClick}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`w-full aspect-[4/5] sm:aspect-[16/9] md:aspect-[21/7] rounded-[2.5rem] overflow-hidden transition-all duration-500 cursor-pointer relative
              ${selectedImage ? 'bg-black' : 'bg-surface-container-high outline-dashed outline-2 outline-outline/30 hover:bg-surface-container hover:scale-[1.01] flex flex-col items-center justify-center p-6 sm:p-8 text-center'}
              ${isDragging ? 'ring-4 ring-primary ring-inset bg-primary/5' : ''}
            `}
          >
            <AnimatePresence mode="wait">
              {selectedImage ? (
                <motion.div 
                  key="preview"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="w-full h-full relative"
                >
                  <img 
                    src={selectedImage} 
                    alt="Preview" 
                    className="w-full h-full object-cover opacity-70 scale-105 group-hover:scale-110 transition-transform duration-[10s] ease-out"
                    referrerPolicy="no-referrer"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent" />
                  
                  {isSearching ? (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-white space-y-6">
                      <motion.div 
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                        className="w-20 h-20 bg-white/20 backdrop-blur-md rounded-full flex items-center justify-center border border-white/30"
                      >
                        <Loader2 size={40} />
                      </motion.div>
                      <div className="space-y-2 text-center">
                        <p className="font-bold tracking-widest uppercase text-sm">{loadingStep}</p>
                        <p className="text-white/60 text-xs">Vui lòng đợi trong giây lát</p>
                      </div>
                    </div>
                  ) : (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-white space-y-4">
                      <div className="w-16 h-16 bg-white/20 backdrop-blur-md rounded-full flex items-center justify-center border border-white/30">
                        <UploadCloud size={32} />
                      </div>
                      <p className="font-bold tracking-widest uppercase text-xs">Nhấp để đổi ảnh</p>
                    </div>
                  )}
                  
                  {!isSearching && (
                    <button 
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedImage(null);
                        setSelectedFile(null);
                        setError(null);
                      }}
                      className="absolute top-6 right-6 w-12 h-12 bg-black/50 hover:bg-red-500 backdrop-blur-md text-white rounded-full flex items-center justify-center transition-all shadow-xl z-20"
                    >
                      <X size={24} />
                    </button>
                  )}
                </motion.div>
              ) : (
                <motion.div 
                  key="upload"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="space-y-6"
                >
                  <div className="w-20 h-20 md:w-24 md:h-24 bg-surface-container-lowest rounded-full flex items-center justify-center shadow-xl mx-auto group-hover:scale-110 transition-transform duration-500">
                    <Camera size={40} className="text-primary fill-primary/10" />
                  </div>
                  <div className="space-y-2">
                    <h3 className="text-xl md:text-2xl font-bold text-on-surface">Thả ảnh món ăn của bạn vào đây</h3>
                    <p className="text-on-surface-variant max-w-xs mx-auto text-sm">Nhấp để chọn hoặc kéo và thả những khám phá thơm ngon của bạn</p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Error Message */}
          <AnimatePresence>
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="mt-4 p-4 bg-red-50 border border-red-200 rounded-2xl text-red-700 text-center text-sm font-medium"
              >
                ⚠️ {error}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Action Button & K Selector */}
          <div className="mt-8 flex flex-col md:flex-row items-center justify-center gap-6">
            <div className={`flex items-center gap-3 bg-surface-container-high px-6 py-4 rounded-2xl transition-opacity duration-300 ${isSearching ? 'opacity-50 pointer-events-none' : 'opacity-100'}`}>
              <Settings2 size={20} className="text-on-surface-variant" />
              <div className="flex flex-col">
                <label className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">
                  Số lượng kết quả (K)
                </label>
                <select 
                  value={kValue} 
                  onChange={(e) => setKValue(Number(e.target.value))}
                  className="bg-transparent text-sm font-bold text-on-surface outline-none cursor-pointer"
                >
                  <option value={5}>5 ảnh (Nhanh)</option>
                  <option value={10}>10 ảnh (Khuyên dùng)</option>
                  <option value={20}>20 ảnh</option>
                  <option value={50}>50 ảnh (Đánh giá sâu)</option>
                </select>
              </div>
            </div>

            <motion.div
              whileHover={selectedImage && !isSearching ? { scale: 1.05 } : {}}
              whileTap={selectedImage && !isSearching ? { scale: 0.95 } : {}}
              className="relative"
            >
              {selectedImage && !isSearching && (
                <motion.div 
                  layoutId="glow"
                  className="absolute -inset-4 bg-primary/20 blur-2xl rounded-full"
                  animate={{ scale: [1, 1.2, 1], opacity: [0.5, 0.8, 0.5] }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
              )}
              <button 
                onClick={handleSearch}
                disabled={!selectedImage || isSearching}
                className={`relative px-10 md:px-12 py-4 md:py-5 font-bold rounded-2xl shadow-xl transition-all duration-500 flex items-center gap-3 group
                  ${selectedImage && !isSearching
                    ? 'bg-gradient-to-r from-primary to-primary-container text-on-primary ring-4 ring-primary/10' 
                    : 'bg-surface-container-highest text-on-surface-variant/40 cursor-not-allowed grayscale'
                  }
                `}
              >
                {isSearching ? (
                  <>
                    <Loader2 size={20} className="animate-spin" />
                    <span>Đang xử lý...</span>
                  </>
                ) : (
                  <>
                    <span>Bắt đầu nhận diện</span>
                    <ChevronRight size={20} className={`transition-transform duration-300 ${selectedImage ? 'group-hover:translate-x-2' : ''}`} />
                  </>
                )}
              </button>
            </motion.div>
          </div>
        </div>
      </motion.section>

      {/* Recent Curations — from search history */}
      <section className="space-y-12">
        <div className="flex flex-col items-center text-center space-y-2">
          <p className="micro-label">Thưởng thức</p>
          <h2 className="text-4xl font-bold">Khám phá gần đây</h2>
        </div>
        
        <div className="flex gap-8 overflow-x-auto pb-6 no-scrollbar snap-x">
          {recentItems.length > 0 ? recentItems.map((item, index) => (
            <motion.div 
              key={item.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + (index * 0.1) }}
              className="flex-shrink-0 w-80 snap-start group"
            >
              <Link 
                to="/detail" 
                state={{
                  dish_name: item.dish_name,
                  similarity: item.similarity,
                  image_url: item.image_url,
                  allResults: item.topResults,
                }}
                className="block space-y-6 text-center"
              >
                <div className="relative aspect-[4/5] rounded-[1.5rem] md:rounded-[2rem] overflow-hidden bg-surface-container-low shadow-lg group-hover:shadow-2xl transition-all duration-500">
                  <img 
                    src={item.image_url} 
                    alt={item.dish_name} 
                    className="w-full h-full object-cover transition-transform duration-1000 group-hover:scale-110"
                    referrerPolicy="no-referrer"
                  />
                  <div className="absolute top-3 right-3 md:top-4 md:right-4">
                    <span className="px-3 md:px-4 py-1.5 bg-primary text-on-primary text-[9px] md:text-[10px] font-bold uppercase tracking-widest rounded-full shadow-lg">
                      {(item.similarity * 100).toFixed(0)}% khớp
                    </span>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="space-y-1">
                    <h3 className="text-2xl font-bold text-on-surface group-hover:text-primary transition-colors">{item.dish_name}</h3>
                    <p className="text-sm font-serif italic text-on-surface-variant">
                      {new Date(item.timestamp).toLocaleDateString('vi-VN', { day: 'numeric', month: 'long', year: 'numeric' })}
                    </p>
                  </div>
                </div>
              </Link>
            </motion.div>
          )) : (
            <div className="w-full py-12 flex flex-col items-center justify-center text-on-surface-variant/40 border-2 border-dashed border-outline/10 rounded-[2rem]">
              <Sparkles size={48} className="mb-4 opacity-20" />
              <p className="font-serif italic italic text-lg text-center px-4">Đang chờ di sản ẩm thực đầu tiên của bạn...</p>
            </div>
          )}
        </div>
        {recentItems.length > 0 && (
          <div className="flex justify-center">
            <Link 
              to="/history"
              className="px-12 py-4 border border-on-surface/10 rounded-full font-bold uppercase text-[10px] tracking-[0.2em] hover:bg-on-surface hover:text-white transition-all"
            >
              Xem tất cả lịch sử
            </Link>
          </div>
        )}
      </section>

      {/* Elegant Closing Quote */}
      <section className="py-24 flex flex-col items-center text-center space-y-8 border-t border-outline/5">
        <div className="w-12 h-px bg-primary/30" />
        <p className="text-3xl md:text-5xl font-serif italic text-on-surface max-w-4xl leading-tight px-6">
          "Ẩm thực là bản giao hưởng của vị giác, nơi mỗi món ăn kể một câu chuyện về vùng đất và con người Việt Nam."
        </p>
        <div className="space-y-2">
          <p className="micro-label !mb-0">Triết lý của chúng tôi</p>
          <div className="w-1 h-1 bg-primary rounded-full mx-auto" />
        </div>
      </section>
    </div>
  );
}
