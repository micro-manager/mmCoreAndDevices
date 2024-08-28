// Comparison of bounded buffers based on different containers.

// Copyright (c) 2003-2008 Jan Gaspar
// Copyright 2013 Paul A. Bristow.  Added some Quickbook snippet markers.

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)
//
// - modified by N.A.
#pragma once

#define BOOST_CB_DISABLE_DEBUG

#include <boost/circular_buffer.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/thread/condition_variable.hpp>
#include <boost/thread/thread.hpp>
#include <boost/call_traits.hpp>
#include <boost/progress.hpp>
#include <boost/bind.hpp>
#include <deque>
#include <list>
#include <string>
#include <iostream>

template <class T>
class bounded_buffer {
public:

    typedef boost::circular_buffer<T> container_type;
    typedef typename container_type::size_type size_type;
    typedef typename container_type::value_type value_type;
    typedef typename boost::call_traits<value_type>::param_type param_type;

    explicit bounded_buffer(size_type capacity) : m_unread(0), m_container(capacity) {}

    void push_front(param_type item) {
        boost::unique_lock<boost::mutex> lock(m_mutex);
        m_not_full.wait(lock, boost::bind(&bounded_buffer<value_type>::is_not_full, this));
        m_container.push_front(item);
        ++m_unread;
        lock.unlock();
        m_not_empty.notify_one();
    }

    void pop_back(value_type* pItem) {
        boost::unique_lock<boost::mutex> lock(m_mutex);
        m_not_empty.wait(lock, boost::bind(&bounded_buffer<value_type>::is_not_empty, this));
        *pItem = m_container[--m_unread];
        lock.unlock();
        m_not_full.notify_one();
    }

    size_type capacity() {return m_container.capacity();}
    size_type size() {return m_container.size();}
    size_type unread() {return m_unread;}
    void clear() {m_container.clear();}
    bool is_not_empty() const { return m_unread > 0; }
    bool is_not_full() const { return m_unread < m_container.capacity(); }

private:
    bounded_buffer(const bounded_buffer&);              // Disabled copy constructor
    bounded_buffer& operator = (const bounded_buffer&); // Disabled assign operator


    size_type m_unread;
    container_type m_container;
    boost::mutex m_mutex;
    boost::condition_variable m_not_empty;
    boost::condition_variable m_not_full;
};

template <class T>
class bounded_buffer_space_optimized {
public:

   typedef boost::circular_buffer_space_optimized<T> container_type;
   typedef typename container_type::size_type size_type;
   typedef typename container_type::value_type value_type;
   typedef typename boost::call_traits<value_type>::param_type param_type;

   explicit bounded_buffer_space_optimized(size_type capacity) : m_container(capacity) {}

   void push_front(param_type item) {
      boost::unique_lock<boost::mutex> lock(m_mutex);
      m_not_full.wait(lock, boost::bind(&bounded_buffer_space_optimized<value_type>::is_not_full, this));
      m_container.push_front(item);
      lock.unlock();
      m_not_empty.notify_one();
   }

   void pop_back(value_type* pItem) {
      boost::unique_lock<boost::mutex> lock(m_mutex);
      m_not_empty.wait(lock, boost::bind(&bounded_buffer_space_optimized<value_type>::is_not_empty, this));
      *pItem = m_container.back();
      m_container.pop_back();
      lock.unlock();
      m_not_full.notify_one();
   }

private:

   bounded_buffer_space_optimized(const bounded_buffer_space_optimized&);              // Disabled copy constructor
   bounded_buffer_space_optimized& operator = (const bounded_buffer_space_optimized&); // Disabled assign operator

   bool is_not_empty() const { return m_container.size() > 0; }
   bool is_not_full() const { return m_container.size() < m_container.capacity(); }

   container_type m_container;
   boost::mutex m_mutex;
   boost::condition_variable m_not_empty;
   boost::condition_variable m_not_full;
};

template <class T>
class bounded_buffer_deque_based {
public:

   typedef std::deque<T> container_type;
   typedef typename container_type::size_type size_type;
   typedef typename container_type::value_type value_type;
   typedef typename boost::call_traits<value_type>::param_type param_type;

   explicit bounded_buffer_deque_based(size_type capacity) : m_capacity(capacity) {}

   void push_front(param_type item) {
      boost::unique_lock<boost::mutex> lock(m_mutex);
      m_not_full.wait(lock, boost::bind(&bounded_buffer_deque_based<value_type>::is_not_full, this));
      m_container.push_front(item);
      lock.unlock();
      m_not_empty.notify_one();
   }

   void pop_back(value_type* pItem) {
      boost::unique_lock<boost::mutex> lock(m_mutex);
      m_not_empty.wait(lock, boost::bind(&bounded_buffer_deque_based<value_type>::is_not_empty, this));
      *pItem = m_container.back();
      m_container.pop_back();
      lock.unlock();
      m_not_full.notify_one();
   }

private:

   bounded_buffer_deque_based(const bounded_buffer_deque_based&);              // Disabled copy constructor
   bounded_buffer_deque_based& operator = (const bounded_buffer_deque_based&); // Disabled assign operator

   bool is_not_empty() const { return m_container.size() > 0; }
   bool is_not_full() const { return m_container.size() < m_capacity; }

   const size_type m_capacity;
   container_type m_container;
   boost::mutex m_mutex;
   boost::condition_variable m_not_empty;
   boost::condition_variable m_not_full;
};

template <class T>
class bounded_buffer_list_based {
public:

   typedef std::list<T> container_type;
   typedef typename container_type::size_type size_type;
   typedef typename container_type::value_type value_type;
   typedef typename boost::call_traits<value_type>::param_type param_type;

   explicit bounded_buffer_list_based(size_type capacity) : m_capacity(capacity) {}

   void push_front(param_type item) {
      boost::unique_lock<boost::mutex> lock(m_mutex);
      m_not_full.wait(lock, boost::bind(&bounded_buffer_list_based<value_type>::is_not_full, this));
      m_container.push_front(item);
      lock.unlock();
      m_not_empty.notify_one();
   }

   void pop_back(value_type* pItem) {
      boost::unique_lock<boost::mutex> lock(m_mutex);
      m_not_empty.wait(lock, boost::bind(&bounded_buffer_list_based<value_type>::is_not_empty, this));
      *pItem = m_container.back();
      m_container.pop_back();
      lock.unlock();
      m_not_full.notify_one();
   }

private:

   bounded_buffer_list_based(const bounded_buffer_list_based&);              // Disabled copy constructor
   bounded_buffer_list_based& operator = (const bounded_buffer_list_based&); // Disabled assign operator

   bool is_not_empty() const { return m_container.size() > 0; }
   bool is_not_full() const { return m_container.size() < m_capacity; }

   const size_type m_capacity;
   container_type m_container;
   boost::mutex m_mutex;
   boost::condition_variable m_not_empty;
   boost::condition_variable m_not_full;
};
