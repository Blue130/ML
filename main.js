import { createApp } from 'vue'
import App from './App.vue'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import router from './router';
import axios from 'axios'; 

const app = createApp(App)

app.use(ElementPlus)
app.use(router); // 确保已经使用 router

// 创建一个 axios 实例  
const http = axios.create({  
  baseURL: 'http://localhost:3000/',  
  // 其他配置...  
}); 
// 将 axios 实例添加到 Vue 应用的原型上  
app.config.globalProperties.$http = http;

import * as ElementPlusIconsVue from '@element-plus/icons-vue'
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

app.mount('#app')