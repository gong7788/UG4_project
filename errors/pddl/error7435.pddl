(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(clear b2)
		(on-table b3)
		(clear b3)
		(on-table b6)
		(clear b6)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(on b8 b9)
		(in-tower b8)
		(on b5 b8)
		(in-tower b5)
		(on b4 b5)
		(in-tower b4)
		(on b0 b4)
		(in-tower b0)
		(on b1 b0)
		(in-tower b1)
		(on b7 b1)
		(in-tower b7)
		(on b2 b7)
		(in-tower b2)
		(red b0)
		(yellow b1)
		(blue b2)
		(red b3)
		(blue b3)
		(blue b4)
		(yellow b5)
		(red b6)
		(green b7)
		(blue b9)
		(green b8)
		(red b2)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?y) (or (not (blue ?y)) (exists (?x) (and (red ?x) (on ?x ?y))))) (and (not (on b7 b8)) (not (on b6 b8)) (not (on b7 b5)) (not (on b6 b5)) (not (on b7 b4)) (not (on b3 b2)))))
)