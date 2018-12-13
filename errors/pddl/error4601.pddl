(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(clear b3)
		(on-table b4)
		(clear b4)
		(on-table b5)
		(clear b5)
		(on-table b6)
		(clear b6)
		(in-tower t0)
		(on b9 t0)
		(in-tower b9)
		(on b8 b9)
		(in-tower b8)
		(on b7 b8)
		(in-tower b7)
		(on b2 b7)
		(in-tower b2)
		(on b0 b2)
		(in-tower b0)
		(on b1 b0)
		(in-tower b1)
		(on b3 b1)
		(in-tower b3)
		(green b0)
		(red b0)
		(green b1)
		(yellow b1)
		(blue b2)
		(green b3)
		(blue b3)
		(yellow b4)
		(green b5)
		(blue b5)
		(red b6)
		(green b8)
		(yellow b8)
		(green b9)
		(yellow b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b6 b7)) (not (on b5 b7)) (not (on b3 b2)) (not (on b4 b3)))))
)