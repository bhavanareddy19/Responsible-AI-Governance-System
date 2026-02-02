import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import Sidebar from '@/components/Sidebar'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
    title: 'AI Governance Dashboard',
    description: 'Responsible AI Governance System for Healthcare',
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en" className="dark">
            <body className={`${inter.className} animated-bg`}>
                <div className="flex min-h-screen">
                    <Sidebar />
                    <main className="flex-1 ml-64 p-8">
                        {children}
                    </main>
                </div>
            </body>
        </html>
    )
}
