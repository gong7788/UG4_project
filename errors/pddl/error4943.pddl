(define (problem block-problem)
	(:domain blocksworld)
	(:objects b0 b1 b2 b3 b4 b5 b6 b7 b8 b9 t0)
	(:init 
		(arm-empty )
		(on-table b3)
		(clear b3)
		(on-table b5)
		(clear b5)
		(on-table b7)
		(clear b7)
		(on-table b8)
		(clear b8)
		(clear b9)
		(in-tower t0)
		(on b0 t0)
		(in-tower b0)
		(on b1 b0)
		(in-tower b1)
		(on b2 b1)
		(in-tower b2)
		(on b4 b2)
		(in-tower b4)
		(on b6 b4)
		(in-tower b6)
		(on b9 b6)
		(in-tower b9)
		(green b3)
		(red b5)
		(maroon b9)
		(blue b3)
	)
	(:goal (and (forall (?x) (in-tower ?x)) (forall (?x) (or (not (red ?x)) (exists (?y) (and (blue ?y) (on ?x ?y))))) (forall (?x) (or (not (green ?x)) (exists (?y) (and (maroon ?y) (on ?x ?y))))) (and (not (on b5 b4)) (not (on b3 b9)))))
)