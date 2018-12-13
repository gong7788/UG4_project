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
		(on-table b7)
		(clear b7)
		(on-table b8)
		(clear b8)
		(on-table b9)
		(clear b9)
		(in-tower t0)
		(on b0 t0)
		(in-tower b0)
		(on b1 b0)
		(in-tower b1)
		(on b2 b1)
		(in-tower b2)
		(on b3 b2)
		(in-tower b3)
		(green b0)
		(yellow b0)
		(green b1)
		(green b2)
		(green b3)
		(yellow b3)
		(blue b4)
		(yellow b5)
		(blue b6)
		(green b7)
		(yellow b7)
		(green b9)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?y) (or (not (yellow ?y)) (exists (?x) (and (green ?x) (on ?x ?y))))) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (and (not (on b5 b3)) (not (on b7 b3)) (not (on b8 b3)))))
)