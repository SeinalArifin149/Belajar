import React from 'react';
import ReactDOM from 'react-dom';

function Home() {
    return (
        <div style={{ fontFamily: 'Arial, sans-serif', textAlign: 'center', padding: '20px' }}>
            <h1>Welcome to My Website</h1>
            <p>This is a simple home page built with React.</p>
            <button 
                style={{
                    padding: '10px 20px',
                    fontSize: '16px',
                    backgroundColor: '#007BFF',
                    color: 'white',
                    border: 'none',
                    borderRadius: '5px',
                    cursor: 'pointer'
                }}
                onClick={() => alert('Button Clicked!')}
            >
                Click Me
            </button>
        </div>
    );
}

ReactDOM.render(<Home />, document.getElementById('root'));// resources/js/Pages/Home.jsx

// import React from 'react';
// import { Head } from '@inertiajs/react';

// export default function Home() {
//   return (
//     <>
//       <Head title="Beranda" />
//       <div className="min-h-screen bg-gray-100 flex items-center justify-center">
//         <div className="bg-white shadow-xl rounded-lg p-10 text-center">
//           <h1 className="text-4xl font-bold text-gray-800 mb-4">Selamat Datang di Aplikasi Kami</h1>
//           <p className="text-gray-600">Ini adalah halaman beranda yang dibuat dengan React + Inertia.js</p>
//         </div>
//       </div>
//     </>
//   );
// }
