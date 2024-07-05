<template>
  <el-table :data="tableData">
    <el-table-column prop="name" label="Name" width="140" />
    <el-table-column prop="age" label="Age" width="120" />
    <el-table-column prop="email" label="Email" />
  </el-table>
</template>

<script>
const tableData = [];
export default {
  name: 'HelloWorld',
  created() {
      this.getUserList();  
    }, 
    data() {  
        return {  
          // 假设我们从API获取的是一个用户列表  
          tableData
        };  
    }, 
    methods: {  
    getUserList() {
      this.$http.get('/').then(response => {  
            console.log(response.data)
            this.tableData = response.data; 
        })  
        .catch(error => {  
            console.log(error)  
        });  
    } 
    }
  
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
h3 {
  margin: 40px 0 0;
}
ul {
  list-style-type: none;
  padding: 0;
}
li {
  display: inline-block;
  margin: 0 10px;
}
a {
  color: #42b983;
}
</style>
