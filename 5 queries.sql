-- =====================================================================
-- DATA SCIENCE FINAL HACKATHON: BUSINESS SQL QUERIES
-- Database: ecommerce_hackathon.db
-- Author: Data Science Hackathon Team
-- Description: 5 Key Business Queries for E-Commerce Customer Intelligence
-- =====================================================================

-- ---------------------------------------------------------------------
-- QUERY 1: Calculate Total Net Revenue
-- Business Objective: Determine overall net revenue generated from all 
-- valid, non-returned transactions after applying discounts.
-- ---------------------------------------------------------------------
SELECT 
    ROUND(SUM(quantity * unit_price * (1 - discount)), 2) AS total_net_revenue
FROM orders
WHERE returned = 0
  AND quantity > 0
  AND unit_price > 0;


-- ---------------------------------------------------------------------
-- QUERY 2: Find Top 10 Customers by Total Spending
-- Business Objective: Identify highest-value VIP customers for targeted 
-- loyalty and retention campaigns.
-- ---------------------------------------------------------------------
SELECT 
    c.customer_name,
    c.city,
    COUNT(o.order_id) AS total_orders,
    ROUND(SUM(o.quantity * o.unit_price * (1 - o.discount)), 2) AS total_spending
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.returned = 0
  AND o.quantity > 0
  AND o.unit_price > 0
GROUP BY c.customer_id, c.customer_name, c.city
ORDER BY total_spending DESC
LIMIT 10;


-- ---------------------------------------------------------------------
-- QUERY 3: Calculate Category-Wise Revenue, Order Count, and Quantity Sold
-- Business Objective: Analyze category contribution to overall revenue 
-- and product movement.
-- ---------------------------------------------------------------------
SELECT 
    TRIM(p.category) AS category,
    ROUND(SUM(o.quantity * o.unit_price * (1 - o.discount)), 2) AS category_revenue,
    COUNT(o.order_id) AS order_count,
    SUM(o.quantity) AS total_quantity_sold
FROM products p
JOIN orders o ON p.product_id = o.product_id
WHERE o.returned = 0
  AND o.quantity > 0
  AND o.unit_price > 0
GROUP BY TRIM(p.category)
ORDER BY category_revenue DESC;


-- ---------------------------------------------------------------------
-- QUERY 4: Calculate Monthly Net Revenue Trend
-- Business Objective: Track sales performance and revenue velocity over 
-- time across monthly intervals.
-- ---------------------------------------------------------------------
SELECT 
    SUBSTR(order_date, 1, 7) AS order_month,
    ROUND(SUM(quantity * unit_price * (1 - discount)), 2) AS monthly_net_revenue,
    COUNT(order_id) AS order_count,
    SUM(quantity) AS quantity_sold
FROM orders
WHERE returned = 0
  AND quantity > 0
  AND unit_price > 0
  AND order_date LIKE '____-__-__'
  AND SUBSTR(order_date, 6, 2) BETWEEN '01' AND '12'
GROUP BY SUBSTR(order_date, 1, 7)
ORDER BY order_month ASC;


-- ---------------------------------------------------------------------
-- QUERY 5: Find Top 5 Products by Net Revenue
-- Business Objective: Identify top revenue-generating individual products 
-- to optimize inventory management and marketing spend.
-- ---------------------------------------------------------------------
SELECT 
    p.product_id,
    p.product_name,
    TRIM(p.category) AS category,
    ROUND(SUM(o.quantity * o.unit_price * (1 - o.discount)), 2) AS net_revenue,
    SUM(o.quantity) AS units_sold
FROM products p
JOIN orders o ON p.product_id = o.product_id
WHERE o.returned = 0
  AND o.quantity > 0
  AND o.unit_price > 0
GROUP BY p.product_id, p.product_name, TRIM(p.category)
ORDER BY net_revenue DESC
LIMIT 5;
